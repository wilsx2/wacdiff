#include "solver.hpp"
#include "initial_conditions.hpp"
#include "timing.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <numeric>
#include <ranges>
#include <utility>
#include <variant>
#include <vector>

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
      _grid_idx{false}, _archive("sim", "temperature", _constants.space_step,
                                 _double_grid[_grid_idx]) {
    std::visit(InitialConditionVisitor{_double_grid[_grid_idx]},
               _constants.initial_conditions);
}

auto SpmdFdmExplicitHeatEqSolver::run() -> void {
    SPDLOG_TRACE("run()");
    auto current_iterations{0u};
    auto converged{false};

    constexpr auto NDims{DistributedStructuredGrid::NDims};
    constexpr auto NumNeighbors{NDims * 2};

    // Compute shell coordinates (adjacent to and depending on halo)
    std::vector<std::array<std::ptrdiff_t, NDims>> shell_coords;
    {
        const auto &local_size{_double_grid[0].local_size()};
        for (auto idx : _double_grid[0].inner_coordinates()) {
            auto on_shell{false};
            for (auto dim = 0; dim < NDims; ++dim) {
                if (idx[dim] == 1 || idx[dim] == local_size[dim]) {
                    on_shell = true;
                    break;
                }
            }
            if (on_shell) {
                shell_coords.push_back(idx);
            }
        }
    }

    if (_constants.storage_interval != 0) {
        _archive.append_state(_double_grid[_grid_idx], current_iterations,
                              current_iterations * _constants.time_step);
    }

    // Perform simulation
    while (!converged) {
        auto &curr{_double_grid[_grid_idx].local_grid()};
        auto &next{_double_grid[!_grid_idx].local_grid()};

        auto apply_stencil{[&](const std::array<std::ptrdiff_t, NDims> &idx) {
            double neighbor_sum{0};
            static constexpr auto DimSeq{std::make_index_sequence<NDims>{}};
            [&]<std::size_t... Ds>(std::index_sequence<Ds...>) {
                ([&] {
                    auto first_neighbor_idx{idx};
                    auto last_neighbor_idx{idx};

                    std::get<Ds>(first_neighbor_idx) -= 1;
                    std::get<Ds>(last_neighbor_idx) += 1;

                    neighbor_sum += curr.apply(first_neighbor_idx) +
                                    curr.apply(last_neighbor_idx);
                }(), ...);
            }(DimSeq);

            auto delta{_gamma *
                      (neighbor_sum -
                        static_cast<double>(NumNeighbors) * curr.apply(idx))};
            next.apply(idx) = curr.apply(idx) + delta;
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
            for (auto idx : _double_grid[_grid_idx].interior_coordinates()) {
                interior_norm_delta += apply_stencil(idx);
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
            for (const auto &idx : shell_coords) {
                shell_norm_delta += apply_stencil(idx);
            }

            auto total_norm_delta{interior_norm_delta + shell_norm_delta};
            const auto inner_count{
                shell_coords.size() +
                _double_grid[_grid_idx].interior_coordinates().size()};

            node_avg_norm_delta =
                total_norm_delta / static_cast<double>(inner_count);
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
                _archive.append_state(_double_grid[_grid_idx], current_iterations,
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
            _archive.append_state(_double_grid[_grid_idx], current_iterations,
                                  current_iterations * _constants.time_step);
        }
    }

    SPDLOG_INFO("Simulation finished: {} iterations, t = {}.",
                current_iterations, current_iterations * _constants.time_step);
}
