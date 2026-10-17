#include "distributed_grid.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

namespace {
auto join_transformed(const auto &range, auto &&transform) -> std::string {
    std::string result;
    bool first{true};
    for (const auto &value : range) {
        if (!first) {
            result += ' ';
        }
        first = false;
        result += transform(value);
    }
    return result;
}

auto repeat_joined(const std::string &value, std::size_t count) -> std::string {
    std::string result;
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) {
            result += ' ';
        }
        result += value;
    }
    return result;
}
} // namespace

void DSGArchive::write_light_data(const DSG &state, int step, double time) {
    auto grid{
        this->_light_file.child("Xdmf").child("Domain").child("Grid").append_child(
            "Grid")};
    grid.append_attribute("Name").set_value("T" + std::to_string(step));
    grid.append_attribute("GridType").set_value("Uniform");

    grid.append_child("Time").append_attribute("Value").set_value(
        std::to_string(time));

    {
        static const std::string point_dimensions_string(
            join_transformed(state.global_size(), [](auto size) {
                return std::format("{}", size + 1);
            }));

        auto topology(grid.append_child("Topology"));
        topology.append_attribute("TopologyType").set_value("3DCoRectMesh");
        topology.append_attribute("Dimensions").set_value(point_dimensions_string);
    }

    {
        auto geometry{grid.append_child("Geometry")};
        geometry.append_attribute("GeometryType").set_value("ORIGIN_DXDYDZ");

        {
            static const std::string ndim_zeros(
                repeat_joined("0.0", static_cast<std::size_t>(NDims)));

            auto origin{geometry.append_child("DataItem")};
            origin.append_attribute("Dimensions").set_value(NDims);
            origin.append_attribute("Format").set_value("XML");
            origin.text().set(ndim_zeros);
        }
        {
            static const std::string ndim_block_sizes(
                repeat_joined(std::format("{}", _block_size),
                              static_cast<std::size_t>(NDims)));
            auto delta{geometry.append_child("DataItem")};
            delta.append_attribute("Dimensions").set_value(NDims);
            delta.append_attribute("Format").set_value("XML");
            delta.text().set(ndim_block_sizes);
        }
    }
    {
        auto attr{grid.append_child("Attribute")};
        attr.append_attribute("Name").set_value(_label);
        attr.append_attribute("AttributeType").set_value("Scalar");
        attr.append_attribute("Center").set_value("Cell");
        {
            static const std::string cell_dimensions_string(
                join_transformed(state.global_size(), [](auto size) {
                    return std::format("{}", size);
                }));

            auto data_item{attr.append_child("DataItem")};
            data_item.append_attribute("Dimensions")
                .set_value(cell_dimensions_string);
            data_item.append_attribute("NumberType").set_value("Float");
            data_item.append_attribute("Precision").set_value(8);
            data_item.append_attribute("Format").set_value("HDF");
            data_item.text().set(
                std::format("{}:/{}", this->_heavy_data_filename, step));
        }
    }
}

void DSGArchive::write_heavy_data(const DSG &state, int step) {
    // https://cvw.cac.cornell.edu/parallel-io-libraries/phdf5/file-operations
    H5Dataspace dataspace{NDims, _cached_global_size.data()};

    auto name{std::to_string(step)};
    H5Dataset dataset{*this->_heavy_file, name.data(), H5T_NATIVE_DOUBLE, *dataspace};

    H5Sselect_hyperslab(*dataspace, H5S_SELECT_SET, _cached_local_start.data(),
                        nullptr, _cached_local_size.data(), nullptr);
    std::array<hsize_t, NDims> interior_start;
    std::fill(interior_start.begin(), interior_start.end(), 1);
    H5Dataspace memoryspace{NDims, _cached_exterior_size.data()};
    H5Sselect_hyperslab(*memoryspace, H5S_SELECT_SET, interior_start.data(),
                        nullptr, _cached_local_size.data(), nullptr);

    auto dxpl{H5Pcreate(H5P_DATASET_XFER)};

    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(*dataset, H5T_NATIVE_DOUBLE, *memoryspace, *dataspace, dxpl,
             state.local_grid().base());

    H5Pclose(dxpl);
}

DSGArchive::DSGArchive(std::string filename, std::string label,
                       double block_size, const DSG &grid)
    : Base(filename, grid.comm(), grid.world_rank()), _label(label),
      _block_size(block_size) {
    std::ranges::transform(grid.global_size(), _cached_global_size.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v); });
    std::ranges::transform(grid.local_start(), _cached_local_start.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v); });
    std::ranges::transform(grid.local_size(), _cached_local_size.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v); });
    std::ranges::transform(grid.local_size(), _cached_exterior_size.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v + 2); });

    if (grid.world_rank() == 0) {
        auto xdmf{this->_light_file.append_child("Xdmf")};
        xdmf.append_attribute("Version").set_value("3.0");

        auto grid_collection{
            xdmf.append_child("Domain").append_child("Grid")};
        grid_collection.append_attribute("Name").set_value("TimeSeries");
        grid_collection.append_attribute("GridType").set_value("Collection");
        grid_collection.append_attribute("CollectionType").set_value("Temporal");
    }
}

void DSGArchive::append_state(const DSG &state, int step, double time) {
    write_heavy_data(state, step);
    if (state.world_rank() == 0) {
        write_light_data(state, step, time);
    }
}
