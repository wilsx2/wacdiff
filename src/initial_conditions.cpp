#include "initial_conditions.hpp"
#include "index_for_each.hpp"

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
    constexpr auto NDims{DistributedStructuredGrid::NDims};
    auto &grid{domain.local_grid()};
    const auto &local_size{domain.local_size()};
    const auto inner_size{domain.inner_size()};

    for (std::size_t linear = 0; linear < inner_size; ++linear) {
        std::array<std::ptrdiff_t, NDims> coordinate{};
        std::ptrdiff_t offset{0};
        std::size_t remaining{linear};
        for (std::ptrdiff_t dim = NDims - 1; dim >= 0; --dim) {
            coordinate[dim] =
                static_cast<std::ptrdiff_t>(
                    remaining % static_cast<std::size_t>(local_size[dim])) +
                1;
            remaining /= static_cast<std::size_t>(local_size[dim]);
            offset += coordinate[dim] * domain.stride(dim);
        }

        static constexpr auto DimSeq{std::make_index_sequence<NDims>{}};
        for_each_index(
            [&](auto D) {
                std::get<decltype(D)::value>(_impl->coordinate) =
                    (static_cast<double>(
                         domain.local_start()[decltype(D)::value]) +
                     static_cast<double>(
                         std::get<decltype(D)::value>(coordinate))) *
                    domain.cell_size();
            },
            DimSeq);
        grid.base()[offset] = _impl->expression.value();
    }
}
