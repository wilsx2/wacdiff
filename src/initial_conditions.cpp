#include "initial_conditions.hpp"
#include "input.hpp"

#include <omp.h>
#include <array>
#include <cstddef>
#include <exprtk.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

ExpressionInitialConditions::ExpressionInitialConditions(std::string_view text)
    : _text(text) {}

auto ExpressionInitialConditions::apply(DistributedStructuredGrid &domain) const -> void {
    auto num_threads    = omp_get_max_threads();
    auto coordinates    = std::vector<std::tuple<double, double, double>>(num_threads);
    auto symbol_tables  = std::vector<exprtk::symbol_table<double>>(num_threads);
    auto expressions    = std::vector<exprtk::expression<double>>(num_threads);
    auto parsers        = std::vector<exprtk::parser<double>>(num_threads);

    #pragma omp parallel
    {
        auto tid = omp_get_thread_num();

        symbol_tables[tid].add_variable("x", std::get<0>(coordinates[tid]));
        symbol_tables[tid].add_variable("y", std::get<1>(coordinates[tid]));
        symbol_tables[tid].add_variable("z", std::get<2>(coordinates[tid]));
        symbol_tables[tid].add_constants();

        expressions[tid].register_symbol_table(symbol_tables[tid]);
        if (!parsers[tid].compile(_text, expressions[tid])) {
            throw std::runtime_error(
                "Arithmetic expression \"" + _text +
                "\" failed to parse: " + parsers[tid].error());
        }
    }

    auto &grid = domain.local_grid();

    auto assign = [&](auto i, auto j, auto k){
        auto tid = omp_get_thread_num();

        std::get<0>(coordinates[tid]) =
            static_cast<double>(domain.local_start()[0] + i - 1)
            * domain.cell_size();
        std::get<1>(coordinates[tid]) =
            static_cast<double>(domain.local_start()[1] + j - 1)
            * domain.cell_size();
        std::get<2>(coordinates[tid]) =
            static_cast<double>(domain.local_start()[2] + k - 1)
            * domain.cell_size();

        grid[i][j][k] = expressions[tid].value();
    };

    #pragma omp parallel
    {
        domain.for_all_in_shell(assign);
        domain.for_all_on_shell(assign);
    }
}
