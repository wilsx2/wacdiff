#pragma once

#include "storage.hpp"
#include <cstdint>
#include <span>

#include "boundary.hpp"
#include "h5raii.hpp"
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <hdf5.h>
#include <mpi.h>
#include <multi/array.hpp>
#include <ranges>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

class DistributedStructuredGrid {
public:
    static constexpr std::ptrdiff_t NDims = 3;

    using AxisBoundary = std::variant<AxisFaces, PeriodicBoundary>;
    using BoundaryConditions = std::span<const AxisBoundary, NDims>;
    using Subarray = decltype(std::declval<boost::multi::array<double, NDims> &>()(
        boost::multi::index_range{}, boost::multi::index_range{}));

private:
    // Data and Partitioning
    std::array<int32_t, NDims> _global_size;
    std::array<int32_t, NDims> _local_size;
    std::array<int32_t, NDims> _local_start;
    std::array<int32_t, NDims> _exterior_size;
    boost::multi::array<double, NDims> _mdarray;
    Subarray _inner_subarray;

    // Metadata
    double _cell_size;

    // Boundary Condition
    std::array<NeumannApplicator<Subarray>, NDims * 2> _neumann_boundaries;
    unsigned _neumann_boundary_count;

    // MPI Information
    int _world_rank, _world_size;
    MPI_Comm _cart_comm;
    int _cart_rank;
    std::array<int, NDims> _cart_coords;
    std::array<int, NDims> _cart_dims;
    std::array<std::pair<int, int>, NDims> _cart_neighbors;
    std::array<std::pair<MPI_Datatype, MPI_Datatype>, NDims>
        _interior_face_types;
    std::array<std::pair<MPI_Datatype, MPI_Datatype>, NDims>
        _exterior_face_types;
    std::array<MPI_Request, NDims * 4> _requests;

    // Methods
    auto init_topology(std::span<int, NDims> periods) -> void;
    auto init_transfer_types() -> void;
    auto deinit_transfer_types() -> void;
    auto init_requests() -> void;
    auto deinit_requests() -> void;
    template <std::size_t... I>
    auto create_subarray(std::array<int, NDims> from, std::array<int, NDims> to,
                         std::index_sequence<I...>);
    auto get_face(std::ptrdiff_t axis, FaceEnd end, FaceKind kind) -> Subarray;
    auto set_local_boundaries(std::span<const AxisBoundary, NDims> axis_boundaries)
        -> void;

public:
    DistributedStructuredGrid(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid &operator=(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid &operator=(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid(
        std::span<const int, NDims> size,
        BoundaryConditions boundary_specification,
        double block_size = 1, double default_value = 0);
    ~DistributedStructuredGrid();
    auto synchronize() -> void;
    auto inner_coordinates() const {
        std::array<std::ptrdiff_t, NDims> sizes{};
        std::size_t total{1};
        for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
            sizes[dim] = static_cast<std::ptrdiff_t>(_exterior_size[dim]) - 2;
            total *= static_cast<std::size_t>(sizes[dim]);
        }

        auto decode = [sizes](std::size_t linear) {
            std::array<std::ptrdiff_t, NDims> coordinate{};
            std::size_t remaining{linear};
            for (std::ptrdiff_t dim = NDims - 1; dim >= 0; --dim) {
                coordinate[dim] =
                    static_cast<std::ptrdiff_t>(
                        remaining % static_cast<std::size_t>(sizes[dim])) +
                    1;
                remaining /= static_cast<std::size_t>(sizes[dim]);
            }
            return coordinate;
        };
        return std::views::iota(std::size_t{0}, total) |
               std::views::transform(decode);
    }
    auto fill(double value) { std::ranges::fill(_inner_subarray.elements(), value); }
    auto local_grid() -> boost::multi::array<double, NDims> & { return _mdarray; }
    auto local_grid() const -> const boost::multi::array<double, NDims> & {
        return _mdarray;
    }
    auto local_size() const -> const std::array<int, NDims> & {
        return _local_size;
    }
    auto local_start() const -> const std::array<int, NDims> & {
        return _local_start;
    }
    auto global_size() const -> const std::array<int, NDims> & {
        return _global_size;
    }
    auto cell_size() const -> double { return _cell_size; }

    auto comm() const -> MPI_Comm { return _cart_comm; }
    auto world_rank() const -> int { return _world_rank; }
};

class DSGArchive : public Archive<DistributedStructuredGrid> {
private:
    using DSG  = DistributedStructuredGrid;
    using Base = Archive<DSG>;

    static constexpr std::ptrdiff_t NDims = DistributedStructuredGrid::NDims;

    std::array<hsize_t, NDims> _cached_global_size;
    std::array<hsize_t, NDims> _cached_local_start;
    std::array<hsize_t, NDims> _cached_local_size;
    std::array<hsize_t, NDims> _cached_exterior_size;
    std::string _label;
    double _block_size;

    void write_light_data(const DSG &state, int step, double time);
    void write_heavy_data(const DSG &state, int step);

public:
    DSGArchive(std::string filename, std::string label, double block_size,
               const DSG &grid);
    void append_state(const DSG &state, int step, double time);
};
