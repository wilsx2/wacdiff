#include "solver.hpp"
#include "index_for_each.hpp"
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
                                             _constants.boundary_conditions,
                                             _constants.space_step},
                   DistributedStructuredGrid{_constants.domain_size,
                                             _constants.boundary_conditions,
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
    constexpr auto NumNeighbors{NDims * 2};

    const auto inner_size{_double_grid[0].inner_size()};
    const auto interior_size{_double_grid[0].interior_size()};
    const auto shell_size{_double_grid[0].shell_size()};

    if (_constants.storage_interval != 0) {
        _archive->append_state(_double_grid[_grid_idx], current_iterations,
                               current_iterations * _constants.time_step);
    }

    // Perform simulation
    while (!converged) {
        auto &curr{_double_grid[_grid_idx].local_grid()};
        auto &next{_double_grid[!_grid_idx].local_grid()};
        auto *curr_data{curr.base()};
        auto *next_data{next.base()};
        const std::array<std::ptrdiff_t, NDims> strides{
            _double_grid[_grid_idx].stride(0),
            _double_grid[_grid_idx].stride(1),
            _double_grid[_grid_idx].stride(2)};

        auto apply_stencil{[&, strides](std::ptrdiff_t center) {
            double neighbor_sum{0};
            static constexpr auto DimSeq{std::make_index_sequence<NDims>{}};
            for_each_index(
                [&](auto D) {
                    constexpr auto dim{decltype(D)::value};
                    const auto stride{strides[dim]};
                    neighbor_sum += curr_data[center - stride] +
                                    curr_data[center + stride];
                },
                DimSeq);

            const auto delta{
                _gamma * (neighbor_sum - static_cast<double>(NumNeighbors) *
                                             curr_data[center])};
            next_data[center] = curr_data[center] + delta;
            return delta * delta;
        }};

        double interior_norm_delta{0.0};
        double node_avg_norm_delta{0.0};
        double global_sum_avg_norm_delta{0.0};
        double avg_norm_delta{0.0};

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            _double_grid[_grid_idx].begin_halo_exchange();

            interior_norm_delta = 0.0;
#pragma omp parallel for reduction(+ : interior_norm_delta)
            for (std::size_t linear = 0; linear < interior_size; ++linear) {
                interior_norm_delta += apply_stencil(
                    _double_grid[_grid_idx].interior_offset(linear));
            }
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Communication);
            #endif

            // Halos are valid now; finish the shell cells that read them.
            _double_grid[_grid_idx].await_halo_exchange();
            assert(!_double_grid[_grid_idx].requests_in_flight());
        }

        {
            #ifdef WACDIFF_PROFILING
            auto timer = ScopedTimer(Phase::Computation);
            #endif

            auto shell_norm_delta{0.0};
#pragma omp parallel for reduction(+ : shell_norm_delta)
            for (std::size_t linear = 0; linear < shell_size; ++linear) {
                shell_norm_delta += apply_stencil(
                    _double_grid[_grid_idx].shell_offset(linear));
            }

            auto total_norm_delta{interior_norm_delta + shell_norm_delta};

            node_avg_norm_delta =
                total_norm_delta / static_cast<double>(inner_size);
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
