#include <filesystem>

#include "distributed_grid.hpp"
#include "initial_conditions.hpp"
#include "input.hpp"
#include "solver.hpp"
#include "timing.hpp"

#include <argparse/argparse.hpp>
#include <mpi.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

template <typename InitialConditions>
auto parse_config(const pugi::xml_node &sim)
    -> SpmdFdmExplicitHeatEqSolver::Configuration {
    using Config = SpmdFdmExplicitHeatEqSolver::Configuration;
    constexpr auto NDims{DistributedStructuredGrid::NDims};
    Config config{};

    config.diffusion_constant = child_value<double>(sim, "diffusion");
    config.storage_interval = child_value<unsigned>(sim, "checkpoint");

    auto grid{required_child(sim, "discretization")};
    config.time_step = child_value<double>(grid, "time_step");
    config.space_step = child_value<double>(grid, "cell_size");
    auto dimensions{required_child(grid, "dimensions")};
    for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
        config.domain_size[dim] =
            child_value<int>(dimensions, "dim" + std::to_string(dim));
    }

    config.initial_conditions =
        InitialConditions{child_text(sim, "initial_conditions")};

    auto convergence{required_child(sim, "convergence_conditions")};
    config.max_iterations =
        child_value<unsigned>(convergence, "iterations_under");
    config.epsilon = child_value<double>(convergence, "avg_delta_over");

    SPDLOG_TRACE("XML input file parsed");
    return config;
}

template <typename InitialConditions>
auto solve(const pugi::xml_node &sim) -> void {
    auto config{parse_config<InitialConditions>(sim)};

    std::unique_ptr<SpmdFdmExplicitHeatEqSolver> solver;
    {
        #ifdef WACDIFF_PROFILING
        auto timer = ScopedTimer(Phase::Computation);
        #endif
        solver = std::make_unique<SpmdFdmExplicitHeatEqSolver>(
            std::move(config));
    }

    solver->run();

    {
        #ifdef WACDIFF_PROFILING
        auto timer = ScopedTimer(Phase::Computation);
        #endif
        solver.reset();
    }
}

inline auto static_dispatch_solve(const pugi::xml_node &sim) -> void {
    auto ic_type{child_attribute_text(sim, "initial_conditions", "type")};
    if (ic_type == "expr") {
        return solve<ExpressionInitialConditions>(sim);
    }

    throw ConfigError{"Unsupported initial condition type \"" +
                      std::string(ic_type) +
                      "\", only \"expr\" is supported"};
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("wac-heat");

    program.add_argument("input")
        .help("XML input file describing the simulation to be performed")
        .required();
    program.add_argument("--output", "-o")
        .help("Path to output simulation data and logs")
        .default_value(".");
    program.add_argument("--log").default_value("err").help(
        "Log level (trace, debug, info, warn, err, critical, off)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return EXIT_FAILURE;
    }

    spdlog::set_level([&]() {
        auto level{program.get<std::string>("--log")};
        if (level == "trace") {
            return spdlog::level::trace;
        } else if (level == "debug") {
            return spdlog::level::debug;
        } else if (level == "info") {
            return spdlog::level::info;
        } else if (level == "warn") {
            return spdlog::level::warn;
        } else if (level == "err") {
            return spdlog::level::err;
        } else if (level == "critical") {
            return spdlog::level::critical;
        }
        return spdlog::level::off;
    }());


    auto input_file{std::filesystem::absolute(
                        program.get<std::string>("input"))
                        .string()};
    auto output_path{program.get<std::string>("--output")};

    SPDLOG_TRACE("Initializing MPI...");
    MPI_Init(&argc, &argv);
    SPDLOG_TRACE("Initialized MPI.");

    int rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Warn once if OpenMP threads are not bound to places. The solver relies
    // on parallel first-touch for NUMA placement, which only works while each
    // thread stays on its core; an unbound team silently puts every page on
    // one NUMA domain.
    if (rank == 0 && omp_get_proc_bind() == omp_proc_bind_false) {
        SPDLOG_WARN(
            "OpenMP threads are not bound (OMP_PROC_BIND is unset); NUMA "
            "first-touch placement will be unreliable. Set OMP_PROC_BIND=close "
            "and OMP_PLACES=cores.");
    }

    if (rank == 0) {
        try {
            std::filesystem::create_directories(output_path);
        } catch (const std::exception &err) {
            std::cerr << "Directory '" << output_path
                      << "' failed to create \n";
            return EXIT_FAILURE;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    try {
        std::filesystem::current_path(output_path);
    } catch (const std::exception &err) {
        std::cerr << "Directory '" << output_path
                  << "' failed to enter \n";
        return EXIT_FAILURE;
    }

    try {
        SPDLOG_TRACE("Parsing XML input file " + input_file + "...");
        pugi::xml_document doc;
        std::ifstream stream{input_file};
        pugi::xml_parse_result result = doc.load(stream);
        if (!result) {
            throw ConfigError{"Failed to load input file \"" + input_file +
                              "\": " + result.description()};
        }

        auto sim{doc.child("simulation")};
        static_dispatch_solve(sim);

        #ifdef WACDIFF_PROFILING
        RunningTimer::instance().save();
        #endif
    } catch (const std::exception &err) {
        SPDLOG_ERROR("{}", err.what());
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return 0;
}
