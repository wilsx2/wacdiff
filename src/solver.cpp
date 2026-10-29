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
        auto total_norm_delta = 0.0;
        auto node_avg_norm_delta = 0.0;
        auto global_avg_norm_delta = 0.0;

        auto &curr = _double_grid[_grid_idx].local_grid();
        auto &next = _double_grid[!_grid_idx].local_grid();

        const auto &dims = _double_grid[_grid_idx].exterior_size();
        const auto gamma = _gamma;
        const double *c = curr.base();
        double *n = next.base();
        const std::ptrdiff_t sj = dims[2];
        const std::ptrdiff_t si =
            static_cast<std::ptrdiff_t>(dims[1]) * dims[2];
        auto apply_stencil = [=](std::ptrdiff_t i, std::ptrdiff_t j,
                                 std::ptrdiff_t k) -> double {
            constexpr auto NumNeighbors = static_cast<double>(NDims * 2);
            const auto idx = i * si + j * sj + k;
            const auto neighbor_sum =
                c[idx + si] + c[idx - si] +
                c[idx + sj] + c[idx - sj] +
                c[idx + 1] + c[idx - 1];
            const auto delta = gamma * (neighbor_sum - NumNeighbors * c[idx]);

            n[idx] = c[idx] + delta;
            return delta * delta;
        };

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            _double_grid[_grid_idx].begin_halo_exchange();

            // Inner computation (does not depend on halo)
            #pragma omp parallel reduction(+ : total_norm_delta)
            {
                total_norm_delta +=
                    _double_grid[_grid_idx].for_all_on_shell(apply_stencil);
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

            #pragma omp parallel reduction(+ : total_norm_delta)
            {
                total_norm_delta +=
                    _double_grid[_grid_idx].for_all_in_shell(apply_stencil);
            }

            const auto inner_elements =
                static_cast<double>(dims[0] - 2) *
                static_cast<double>(dims[1] - 2) *
                static_cast<double>(dims[2] - 2);
            node_avg_norm_delta = total_norm_delta / inner_elements;
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Communication);
            #endif

            MPI_Allreduce(&node_avg_norm_delta, &global_avg_norm_delta, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            global_avg_norm_delta /= _double_grid[_grid_idx].world_size();

            // Swap roles of current and next grids
            _grid_idx = !_grid_idx;
            ++current_iterations;

            // Check convergence
            converged = global_avg_norm_delta < _constants.epsilon ||
                        current_iterations >= _constants.max_iterations;
        }

        SPDLOG_TRACE("current_iterations={}, avg_norm_delta={}",
                     current_iterations, global_avg_norm_delta);

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
