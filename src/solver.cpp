#include "solver.hpp"
#include "initial_conditions.hpp"
#include "timing.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <utility>
#include <variant>

SpmdFdmExplicitHeatEqSolver::SpmdFdmExplicitHeatEqSolver(Configuration &&config)
    : _constants{std::move(config)},
      _gamma(_constants.diffusion_constant *
             (_constants.time_step /
              (_constants.space_step * _constants.space_step))),
      _double_grid{DistributedStructuredGrid{_constants.domain_size,
                                             _constants.space_step},
                   DistributedStructuredGrid{_constants.domain_size,
                                             _constants.space_step}},
      _grid_idx{false} {
    std::visit(InitialConditionVisitor{_double_grid[_grid_idx]},
               _constants.initial_conditions);

    if (_constants.storage_interval != 0) {
        _archive.emplace("sim", "temperature", _constants.space_step,
                         _double_grid[_grid_idx]);
    }
}

auto SpmdFdmExplicitHeatEqSolver::run() -> void {
    SPDLOG_TRACE("run()");
    auto current_iterations{0u};
    auto converged{false};

    constexpr auto NDims{DistributedStructuredGrid::NDims};

    if (_constants.storage_interval != 0) {
        _archive->append_state(_double_grid[_grid_idx], current_iterations,
                               current_iterations * _constants.time_step);
    }

    // Perform simulation
    while (!converged) {
        auto &curr = _double_grid[_grid_idx].local_grid();
        auto &next = _double_grid[!_grid_idx].local_grid();
        const auto &dims = _double_grid[_grid_idx].exterior_size();
        auto apply_stencil = [&](int i, int j, int k) {
            constexpr auto NumNeighbors = static_cast<double>(NDims * 2);
            const auto neighbor_sum =
                curr[i + 1][j][k] + curr[i - 1][j][k] +
                curr[i][j + 1][k] + curr[i][j - 1][k] +
                curr[i][j][k + 1] + curr[i][j][k - 1];
            const auto delta =
                _gamma * (neighbor_sum - NumNeighbors * curr[i][j][k]);

            next[i][j][k] = curr[i][j][k] + delta;
            return delta * delta;
        };

        auto total_norm_delta = 0.0;
        auto node_avg_norm_delta = 0.0;
        auto global_sum_avg_norm_delta = 0.0;
        auto avg_norm_delta = 0.0;

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            _double_grid[_grid_idx].begin_halo_exchange();

            total_norm_delta = 0.0;
            // TODO: vectorize
            // Inner computation (does not depend on halo)
            #pragma omp parallel for schedule(static) collapse(3) reduction(+ : total_norm_delta)
            for (int i = 2; i < dims[0] - 2; i++) {
                for (int j = 2; j < dims[1] - 2; j++) {
                    for (int k = 2; k < dims[2] - 2; k++) {
                        total_norm_delta += apply_stencil(i, j, k);
                    }
                }
            }
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Communication);
            #endif

            _double_grid[_grid_idx].await_halo_exchange();
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            // Shell computation (disjoint face/edge decomposition)
            // x-faces: full y-z range, including y/z boundaries
            #pragma omp parallel for schedule(static) collapse(2) reduction(+ : total_norm_delta)
            for (int j = 1; j < dims[1] - 1; j++) {
                for (int k = 1; k < dims[2] - 1; k++) {
                    total_norm_delta += apply_stencil(1, j, k);
                    total_norm_delta += apply_stencil(dims[0] - 2, j, k);
                }
            }
            // y-faces: exclude x-boundary cells already handled above
            #pragma omp parallel for schedule(static) collapse(2) reduction(+ : total_norm_delta)
            for (int i = 2; i < dims[0] - 2; i++) {
                for (int k = 1; k < dims[2] - 1; k++) {
                    total_norm_delta += apply_stencil(i, 1, k);
                    total_norm_delta += apply_stencil(i, dims[1] - 2, k);
                }
            }
            // z-faces: exclude x- and y-boundary cells already handled above
            #pragma omp parallel for schedule(static) collapse(2) reduction(+ : total_norm_delta)
            for (int i = 2; i < dims[0] - 2; i++) {
                for (int j = 2; j < dims[1] - 2; j++) {
                    total_norm_delta += apply_stencil(i, j, 1);
                    total_norm_delta += apply_stencil(i, j, dims[2] - 2);
                }
            }

            const auto inner_size = static_cast<double>(dims[0] - 2) *
                                    static_cast<double>(dims[1] - 2) *
                                    static_cast<double>(dims[2] - 2);
            node_avg_norm_delta = total_norm_delta / inner_size;
            global_sum_avg_norm_delta = 0.0;
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Communication);
            #endif

            MPI_Allreduce(&node_avg_norm_delta, &global_sum_avg_norm_delta, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            avg_norm_delta = global_sum_avg_norm_delta /
                             _double_grid[_grid_idx].world_size();

            // Swap roles of current and next grids
            _grid_idx = !_grid_idx;
            ++current_iterations;

            // Check convergence
            converged = avg_norm_delta < _constants.epsilon ||
                        current_iterations >= _constants.max_iterations;
        }

        SPDLOG_TRACE("current_iterations={}, avg_norm_delta={}",
                     current_iterations, avg_norm_delta);

        // Save checkpoint
        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::IO);
            #endif

            if (_constants.storage_interval != 0 &&
                current_iterations % _constants.storage_interval == 0) {
                _archive->append_state(_double_grid[_grid_idx], current_iterations,
                                       current_iterations * _constants.time_step);
            }
        }
    }
    {
        #ifdef WACDIFF_PROFILING
        auto timer = ScopedTimer(Phase::IO);
        #endif

        // Make sure the final state is stored
        if (_constants.storage_interval != 0 &&
            current_iterations % _constants.storage_interval != 0) {
            _archive->append_state(_double_grid[_grid_idx], current_iterations,
                                   current_iterations * _constants.time_step);
        }
    }

    SPDLOG_INFO("Simulation finished: {} iterations, t = {}.",
                current_iterations, current_iterations * _constants.time_step);
}
