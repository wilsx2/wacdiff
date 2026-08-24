#pragma once

#include <spdlog/spdlog.h>
#include <multi/array.hpp>

struct StabilityAndIterationConditions {
    double epsilon;
    unsigned max_iterations;
    unsigned current_iterations;

    auto operator()(const boost::multi::array<double, 2> &curr,
                    boost::multi::array<double, 2> &next) {
        auto total_norm_delta{0.0};
        # pragma omp parallel for reduction(+:total_norm_delta)
        for (auto [i, j] : curr.extents().elements()) {
            auto delta{curr[i][j] - next[i][j]};
            total_norm_delta += delta * delta;
        }
        auto avg_norm_delta{total_norm_delta / static_cast<double>(curr.size())};
        SPDLOG_TRACE("avg_norm_delta: {}", avg_norm_delta);
        return avg_norm_delta < epsilon ||
               current_iterations++ >= max_iterations;
    }
};
