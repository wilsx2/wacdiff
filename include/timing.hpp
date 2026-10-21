#pragma once
#ifdef WACDIFF_PROFILING

#include <pugixml.hpp>
#include <array>
#include <chrono>

using Duration = std::chrono::duration<double, std::milli>;
enum class Phase : char { Compute = 0, Communication = 1, IO = 2 };

class RunningTimer {
    private:
    std::array<Duration, 3> _durations;
    RunningTimer();
    ~RunningTimer();

    public:
    RunningTimer(const RunningTimer&) = delete;
    RunningTimer& operator=(const RunningTimer&) = delete;

    static RunningTimer& instance() {
        static RunningTimer singleton;
        return singleton;
    }
    auto& at(Phase phase){
        return _durations[static_cast<char>(phase)];
    }
    auto save() -> void {
        // Reduce to root
        int root = 0;
        int rank, world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == root) {
            MPI_Reduce(MPI_INPLACE, _durations.data(), 3, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(_durations.data(), NULL, 3, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
        }
        for (auto& val : _durations) {
            val /= world_size;
        }

        // Construct report
        auto xml = pugi::xml_document{};

        using std::to_string;
        xml.append_child("units").set_value("ms");
        xml.append_child("compute")       .set_value(to_string(_durations[0]));
        xml.append_child("communication") .set_value(to_string(_durations[1]));
        xml.append_child("io")            .set_value(to_string(_durations[2]));

        auto success = xml.save_file("timing.xml");
        if (!success) {
            SPDLOG_ERROR("Timing report failed to save");
        }
    }
};

class ScopedTimer {
    private:
    Phase _phase;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        _start;
    
    public:
    ScopedTimer(Phase phase) {
        _phase = phase;
        _start_time = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        Duration duration = end - start;
        RunningTimer::instance().at(phase) += duration;
    }
};

#endif
