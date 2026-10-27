#pragma once
#include "distributed_grid.hpp"
#include <cstddef>
#include <memory>
#include <string_view>
#include <variant>

class InitialConditionVisitor;

class ConstantInitialConditions {
    friend InitialConditionVisitor;

private:
    double _value;

public:
    ConstantInitialConditions() = default;
    ConstantInitialConditions(std::string_view text);
};

class ExpressionInitialConditions {
    friend InitialConditionVisitor;

private:
    std::string _text;
    auto apply(DistributedStructuredGrid &domain) const -> void;

public:
    ExpressionInitialConditions() = default;
    ExpressionInitialConditions(std::string_view text);
};

using InitialConditionUnion =
    std::variant<ConstantInitialConditions, ExpressionInitialConditions>;

class InitialConditionVisitor {
private:
    DistributedStructuredGrid &_domain;

public:
    InitialConditionVisitor(DistributedStructuredGrid &domain)
        : _domain(domain) {}
    void operator()(const ConstantInitialConditions &ic) {
        _domain.fill(ic._value);
    }
    void operator()(const ExpressionInitialConditions &ic) {
        ic.apply(_domain);
    }
};
