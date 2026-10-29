#pragma once

#include "distributed_grid.hpp"

#include <hdf5.h>
#include <mpi.h>
#include <pugixml.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

struct ArchiveOptions {
    hsize_t alignment{1 << 20};
    std::vector<std::pair<std::string, std::string>> hints{
        {"romio_ds_write", "disable"},
    };
};

class Archive {
private:
    using DSG = DistributedStructuredGrid;

    static constexpr std::ptrdiff_t NDims = DSG::NDims;

    int _rank;
    std::string _light_data_filename;
    std::string _heavy_data_filename;
    pugi::xml_document _light_file;
    hid_t _heavy_file;


    std::array<hsize_t, NDims> _cached_global_size;
    std::array<hsize_t, NDims> _cached_local_start;
    std::array<hsize_t, NDims> _cached_local_size;
    std::array<hsize_t, NDims> _cached_exterior_size;
    std::array<hsize_t, NDims> _cached_chunk_size;
    std::string _label;
    double _block_size;

    void write_light_data(const DSG &state, int step, double time);
    void write_heavy_data(const DSG &state, int step);

public:
    Archive(const std::string &file_prefix, std::string label,
            double block_size, const DSG &grid,
            const ArchiveOptions &options);
    ~Archive();
    Archive(const Archive &) = delete;
    Archive &operator=(const Archive &) = delete;

    void append_state(const DSG &state, int step, double time);
};
