#pragma once

#include "distributed_grid.hpp"
#include "initial_conditions.hpp"

#include <mpi.h>
#include <spdlog/spdlog.h>

#include <array>
#include <optional>

class SpmdFdmExplicitHeatEqSolver {
public:
    struct Configuration {
        std::array<int, DistributedStructuredGrid::NDims> domain_size;
        // Heat Equation Parameters
        double diffusion_constant;
        double time_step;
        double space_step;
        // Initial Conditions
        InitialConditionUnion initial_conditions;
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

    std::optional<DSGArchive> _archive;

public:
    SpmdFdmExplicitHeatEqSolver() = delete;
    SpmdFdmExplicitHeatEqSolver(Configuration &&config);
    auto run() -> void;
};
