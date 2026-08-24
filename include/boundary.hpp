#pragma once

#include <multi/array.hpp>

struct DierichletBoundary {
    double value;
};

struct NeumannBoundary {
    double delta;
};

using NonPeriodicBoundary = std::variant<DierichletBoundary, NeumannBoundary>;

struct PeriodicBoundary {};
