#pragma once

#include "distributed_grid.hpp"
#include "h5raii.hpp"
#include "initial_conditions.hpp"
#include "storage.hpp"

#include <mpi.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

template <std::invocable<DistributedStructuredGrid &> InitialConditions>
class SpmdFdmExplicitHeatEqSolver {
public:
    struct Configuration {
        std::array<int, DistributedStructuredGrid::NDims> domain_size;
        // Heat Equation Parameters
        double diffusion_constant;
        double time_step;
        double space_step;
        // Initial Conditions
        InitialConditions initial_conditions;
        // Boundary Conditions
        std::array<DistributedStructuredGrid::AxisBoundary,
                   DistributedStructuredGrid::NDims>
            boundary_conditions;
        // Convergence Params
        double epsilon;
        unsigned max_iterations;
        // Storage Params
        unsigned storage_interval;
    };

private:
    const Configuration _constants;
    double _gamma;
    std::array<DistributedStructuredGrid, 2> _double_grid;
    bool _grid_idx;

    DSGArchive _archive;

public:
    SpmdFdmExplicitHeatEqSolver() = delete;
    SpmdFdmExplicitHeatEqSolver(Configuration &&config);
    auto run() -> void;
};

extern template class SpmdFdmExplicitHeatEqSolver<ConstantInitialConditions>;
extern template class SpmdFdmExplicitHeatEqSolver<ExpressionInitialConditions>;
