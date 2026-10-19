#include "solver.hpp"
#include "initial_conditions.hpp"

#include <algorithm>
#include <array>
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

    if (_constants.storage_interval != 0) {
        _archive.append_state(_double_grid[_grid_idx], current_iterations,
                              current_iterations * _constants.time_step);
    }

    // Perform simulation
    while (!converged) {

        // Conditionally perform halo exchange / apply boundary condition
        _double_grid[_grid_idx].synchronize();

        // Solve
        double total_norm_delta{0};
        auto &curr{_double_grid[_grid_idx].local_grid()};
        auto &next{_double_grid[!_grid_idx].local_grid()};
        auto inner_coordinates{_double_grid[_grid_idx].inner_coordinates()};
        constexpr auto NDims{DistributedStructuredGrid::NDims};
#pragma omp parallel for reduction(+ : total_norm_delta)
        for (auto idx : inner_coordinates) {
            constexpr auto NumNeighbors{NDims * 2};
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
            total_norm_delta += delta * delta;
        }

        auto node_avg_norm_delta
            {total_norm_delta / static_cast<double>(inner_coordinates.size())};
        decltype(node_avg_norm_delta) global_sum_avg_norm_delta;
        MPI_Allreduce(
            &global_sum_avg_norm_delta,
            &node_avg_norm_delta,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        auto avg_norm_delta
            {global_sum_avg_norm_delta / _double_grid[_grid_idx].world_rank()};

        // Swap roles of current and next grids
        _grid_idx = !_grid_idx;
        ++current_iterations;

        // Check convergence
        converged = avg_norm_delta < _constants.epsilon ||
                    current_iterations >= _constants.max_iterations;
        SPDLOG_TRACE("current_iterations={}, avg_norm_delta={}",
                     current_iterations, avg_norm_delta);

        // Save checkpoint
        if (_constants.storage_interval != 0 &&
            current_iterations % _constants.storage_interval == 0) {
            _archive.append_state(_double_grid[_grid_idx], current_iterations,
                                  current_iterations * _constants.time_step);
        }
    }
    // Make sure the final state is stored, unless it already landed on a
    // checkpoint (keeping dataset names unique).
    if (_constants.storage_interval != 0 &&
        current_iterations % _constants.storage_interval != 0) {
        _archive.append_state(_double_grid[_grid_idx], current_iterations,
                              current_iterations * _constants.time_step);
    }

    SPDLOG_INFO("Simulation finished: {} iterations, t = {}.",
                current_iterations, current_iterations * _constants.time_step);
}
