#pragma once

#include <multi/array.hpp>

// https://en.wikipedia.org/wiki/Finite_difference_method#Explicit_method
struct FDMExplicitHeat2D {
    double gamma;

    FDMExplicitHeat2D(double diffusion_factor, double space_step, double time_step)
        : gamma(diffusion_factor * (time_step / (space_step * space_step))) {}
    auto operator()(const boost::multi::array<double, 2> &curr,
                    boost::multi::array<double, 2> &next) const {
        #pragma omp parallel for
        for (auto [i, j] : curr.extents().elements()) {
            auto neighbor_sum{curr[i + 1][j] + curr[i - 1][j] + curr[i][j + 1] +
                              curr[i][j - 1]};
            auto delta{neighbor_sum - 4.0 * curr[i][j]};
            next[i][j] = curr[i][j] + gamma * delta;
        }
    }
};

/*
// https://people.sc.fsu.edu/~jpeterson/4-Implicit.pdf
struct FDMImplicitHeat2D {
    double gamma;

    FDMImplicitHeat2D(double diffusion_factor, double space_step, double time_step)
        : gamma(diffusion_factor * (time_step / (space_step * space_step))) {}
};
*/
