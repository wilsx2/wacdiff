#pragma once

struct DirichletBoundary {
    double value;
};

struct NeumannBoundary {
    double delta;
};

using NonPeriodicBoundary = std::variant<DirichletBoundary, NeumannBoundary>;

struct AxisFaces {
    NonPeriodicBoundary first;
    NonPeriodicBoundary last;
};

struct PeriodicBoundary {};

enum class FaceEnd  { First, Last };
enum class FaceKind { Interior, Exterior };

template <std::ranges::range Region>
struct DirichletApplicator {
    double value;
    Region region;
    auto apply() -> void {
        auto elements{region.elements()};
#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < elements.size(); ++i) {
            elements[i] = value;
        }
    }
};

template <std::ranges::range Region>
struct NeumannApplicator {
    double delta;
    Region target;
    Region source;
    auto apply() -> void {
        auto targets{target.elements()};
        auto sources{source.elements()};
#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < targets.size(); ++i) {
            targets[i] = sources[i] + delta;
        }
    }
};
