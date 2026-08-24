#pragma once
#include "distributed_grid.hpp"
#include <cstddef>
#include <memory>

class ConstantInitialConditions {
private:
    double _value;

public:
    ConstantInitialConditions() = default;
    ConstantInitialConditions(std::string_view text);
    template <typename T>
    auto operator()(T &domain) const {
        domain.fill(_value);
    }
};

class ExpressionInitialConditions {
private:
    struct Impl;
    std::unique_ptr<Impl> _impl;

    void compile();

public:
    ExpressionInitialConditions();
    ExpressionInitialConditions(std::string_view text);
    ExpressionInitialConditions(const ExpressionInitialConditions &);
    ExpressionInitialConditions(ExpressionInitialConditions &&) noexcept;
    ExpressionInitialConditions &operator=(const ExpressionInitialConditions &);
    ExpressionInitialConditions &operator=(ExpressionInitialConditions &&) noexcept;
    ~ExpressionInitialConditions();
    auto operator()(DistributedStructuredGrid &domain) const -> void;
};
