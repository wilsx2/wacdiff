#pragma once

#include <array>
#include <chrono>
#include <string>

#include <mpi.h>
#include <pugixml.hpp>
#include <spdlog/spdlog.h>

#ifdef WACDIFF_PROFILING

using Duration = std::chrono::duration<double, std::milli>;
enum class Phase : char { Computation = 0, Communication = 1, IO = 2 };

class RunningTimer {
private:
    std::array<Duration, 3> _durations{};

    RunningTimer() = default;
    ~RunningTimer() = default;

public:
    RunningTimer(const RunningTimer &) = delete;
    RunningTimer &operator=(const RunningTimer &) = delete;

    static RunningTimer &instance() {
        static RunningTimer singleton;
        return singleton;
    }

    auto &at(Phase phase) {
        return _durations[static_cast<char>(phase)];
    }

    auto save() -> void {
        constexpr int NumPhases = 3;
        constexpr int root = 0;

        std::array<double, NumPhases> values{};
        for (auto i = 0; i < NumPhases; ++i) {
            values[i] = _durations[i].count();
        }

        int rank{};
        int world_size{};
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        std::array<double, NumPhases> reduced{};
        MPI_Reduce(values.data(), reduced.data(), NumPhases, MPI_DOUBLE,
                   MPI_SUM, root, MPI_COMM_WORLD);

        if (rank != root) {
            return;
        }

        for (auto &value : reduced) {
            value /= static_cast<double>(world_size);
        }

        pugi::xml_document xml{};

        auto append_text{[](pugi::xml_node node, const std::string &value) {
            node.append_child(pugi::node_pcdata).set_value(value.c_str());
        }};

        append_text(xml.append_child("units"), "ms");
        append_text(xml.append_child("computation"),
                    std::to_string(reduced[0]));
        append_text(xml.append_child("communication"),
                    std::to_string(reduced[1]));
        append_text(xml.append_child("io"), std::to_string(reduced[2]));

        if (!xml.save_file("timing.xml")) {
            SPDLOG_ERROR("Timing report failed to save");
        }
    }
};

class ScopedTimer {
private:
    Phase _phase;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;

public:
    ScopedTimer(Phase phase)
        : _phase{phase},
          _start{std::chrono::high_resolution_clock::now()} {}

    ~ScopedTimer() {
        auto end{std::chrono::high_resolution_clock::now()};
        RunningTimer::instance().at(_phase) += end - _start;
    }
};

#endif
