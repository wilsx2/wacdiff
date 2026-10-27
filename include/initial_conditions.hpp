#pragma once
#include "distributed_grid.hpp"
#include <cstddef>
#include <memory>
#include <string_view>
#include <variant>

class InitialConditionVisitor;

class ExpressionInitialConditions {
    friend InitialConditionVisitor;

private:
    std::string _text;
    auto apply(DistributedStructuredGrid &domain) const -> void;

public:
    ExpressionInitialConditions() = default;
    ExpressionInitialConditions(std::string_view text);
};

using InitialConditionUnion = std::variant<ExpressionInitialConditions>;

class InitialConditionVisitor {
private:
    DistributedStructuredGrid &_domain;

public:
    InitialConditionVisitor(DistributedStructuredGrid &domain)
        : _domain(domain) {}
    void operator()(const ExpressionInitialConditions &ic) {
        ic.apply(_domain);
    }
};
