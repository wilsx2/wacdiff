#pragma once

#include "distributed_grid.hpp"
#include "domain.hpp"
#include <concepts>
#include <cstddef>
#include <multi/array.hpp>
#include <ranges>
#include <variant>

// General
template <typename T>
concept Constraint = std::invocable<T>;

template <std::ranges::range Region>
struct PinValue {
    Region region;
    double value;
};

template <std::ranges::range Region>
struct PinDelta {
    Region target;
    Region source;
    double delta;
};

enum class ConstraintKind { PinValue, PinDelta };
template <std::ranges::range Region>
using RuntimeConstraint =
    std::variant<PinValue<Region>, PinDelta<Region>>;

template <std::ranges::range Region>
struct ConstraintVisitor {
    void operator()(PinValue<Region> &constraint) const {
        auto elements{constraint.region.elements()};
#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < elements.size(); ++i) {
            elements[i] = constraint.value;
        }
    }

    void operator()(PinDelta<Region> &constraint) const {
        auto targets{constraint.target.elements()};
        auto sources{constraint.source.elements()};
#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < targets.size(); ++i) {
            targets[i] = sources[i] + constraint.delta;
        }
    }
};

// Type Trait Specialization

struct FaceSelector {
    int axis;
    bool first;
};

struct ConstraintSpec {
    FaceSelector target;
    ConstraintKind kind;
    double value;
};
