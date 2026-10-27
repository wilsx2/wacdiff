#include "initial_conditions.hpp"

#include "input.hpp"
#include <array>
#include <cstddef>
#include <exprtk.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

ConstantInitialConditions::ConstantInitialConditions(std::string_view text) {
    _value = parse_value<double>(text);
}

struct ExpressionInitialConditions::Impl {
    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double>   expression;
    exprtk::parser<double>       parser;
    std::string                  expression_text;
    mutable std::tuple<double, double, double> coordinate{0, 0, 0};
};

void ExpressionInitialConditions::compile() {
    _impl->symbol_table.clear();
    _impl->expression.release();

    _impl->symbol_table.add_variable("x", std::get<0>(_impl->coordinate));
    _impl->symbol_table.add_variable("y", std::get<1>(_impl->coordinate));
    _impl->symbol_table.add_variable("z", std::get<2>(_impl->coordinate));
    _impl->symbol_table.add_constants();

    _impl->expression.register_symbol_table(_impl->symbol_table);

    if (!_impl->parser.compile(_impl->expression_text, _impl->expression)) {
        throw std::runtime_error("Arithmetic expression \"" +
                                 _impl->expression_text +
                                 "\" failed to parse: " +
                                 _impl->parser.error());
    }
}

ExpressionInitialConditions::ExpressionInitialConditions()
    : _impl(std::make_unique<Impl>()) {}

ExpressionInitialConditions::ExpressionInitialConditions(std::string_view text)
    : _impl(std::make_unique<Impl>()) {
    _impl->expression_text = text;
    compile();
}

ExpressionInitialConditions::ExpressionInitialConditions(
    const ExpressionInitialConditions &other)
    : _impl(std::make_unique<Impl>()) {
    _impl->expression_text = other._impl->expression_text;
    compile();
}

ExpressionInitialConditions::ExpressionInitialConditions(
    ExpressionInitialConditions &&other) noexcept = default;

ExpressionInitialConditions &ExpressionInitialConditions::operator=(
    const ExpressionInitialConditions &other) {
    if (this != &other) {
        _impl->expression_text = other._impl->expression_text;
        compile();
    }
    return *this;
}

ExpressionInitialConditions &ExpressionInitialConditions::operator=(
    ExpressionInitialConditions &&other) noexcept = default;

ExpressionInitialConditions::~ExpressionInitialConditions() = default;

auto ExpressionInitialConditions::apply(DistributedStructuredGrid &domain) const -> void {
    auto &grid{domain.local_grid()};
    const auto &local_size{domain.local_size()};

    // WARN: NUMA! Threads have to compete over expression
    //       They should have a copy
    #pragma omp parallel for schedule(static) collapse(3)
    for (std::ptrdiff_t i = 0; i < local_size[0]; ++i) {
        for (std::ptrdiff_t j = 0; j < local_size[1]; ++j) {
            for (std::ptrdiff_t k = 0; k < local_size[2]; ++k) {
                std::get<0>(_impl->coordinate) =
                    static_cast<double>(domain.local_start()[0] + i + 1) *
                    domain.cell_size();
                std::get<1>(_impl->coordinate) =
                    static_cast<double>(domain.local_start()[1] + j + 1) *
                    domain.cell_size();
                std::get<2>(_impl->coordinate) =
                    static_cast<double>(domain.local_start()[2] + k + 1) *
                    domain.cell_size();

                grid[i + 1][j + 1][k + 1] = _impl->expression.value();
            }
        }
    }
}
