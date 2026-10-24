#pragma once

#include "storage.hpp"
#include <cstdint>

#include "boundary.hpp"
#include "h5raii.hpp"
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <hdf5.h>
#include <mpi.h>
#include <multi/array.hpp>
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
    using BoundaryConditions = const std::array<AxisBoundary, NDims> &;
    using Subarray = decltype(std::declval<boost::multi::array<double, NDims> &>()(
        boost::multi::index_range{}, boost::multi::index_range{}));

private:
    // Data and Partitioning
    std::array<int32_t, NDims> _global_size;
    std::array<int32_t, NDims> _local_size;
    std::array<int32_t, NDims> _local_start;
    std::array<int32_t, NDims> _exterior_size;
    std::array<std::ptrdiff_t, NDims> _strides;
    boost::multi::array<double, NDims> _mdarray;

    // Metadata
    double _cell_size;

    // Boundary Condition
    struct NeumannFace {
        double delta;
        int axis;
        FaceEnd end;
    };
    std::array<NeumannFace, NDims * 2> _neumann_boundaries;
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
    bool _requests_in_flight;

    // Methods
    auto init_topology(std::array<int, NDims> &periods) -> void;
    auto init_transfer_types() -> void;
    auto deinit_transfer_types() -> void;
    auto init_requests() -> void;
    auto deinit_requests() -> void;
    template <std::size_t... I>
    auto create_subarray(std::array<int, NDims> from, std::array<int, NDims> to,
                         std::index_sequence<I...>) -> Subarray;
    auto get_face(std::ptrdiff_t axis, FaceEnd end, FaceKind kind) -> Subarray;
    auto inner_subarray() -> Subarray;
    auto set_local_boundaries(
        const std::array<AxisBoundary, NDims> &axis_boundaries) -> void;

public:
    DistributedStructuredGrid(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid &operator=(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid &operator=(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid(
        const std::array<int, NDims> &size,
        BoundaryConditions boundary_specification,
        double block_size = 1, double default_value = 0);
    ~DistributedStructuredGrid();
    auto begin_halo_exchange() -> void;
    auto await_halo_exchange() -> void;
    auto requests_in_flight() -> bool {
        return _requests_in_flight;
    }
    auto inner_size() const -> std::size_t {
        std::size_t total{1};
        for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
            total *= static_cast<std::size_t>(_local_size[dim]);
        }
        return total;
    }
    /// Number of interior cells (indices 2 .. local_size-1). This is zero
    /// when any local dimension is smaller than 3.
    auto interior_size() const -> std::size_t {
        std::size_t total{1};
        for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
            total *= static_cast<std::size_t>(
                std::max(0, _local_size[dim] - 2));
        }
        return total;
    }
    /// Stride, in elements, along one logical dimension of the local grid.
    auto stride(std::ptrdiff_t dim) const -> std::ptrdiff_t {
        return _strides[dim];
    }
    /// Maps a flat index in [0, inner_size()) to an offset into
    /// local_grid().base() for the corresponding inner cell.
    auto inner_offset(std::size_t linear) const -> std::ptrdiff_t {
        std::ptrdiff_t offset{0};
        std::size_t remaining{linear};
        for (std::ptrdiff_t dim = NDims - 1; dim >= 0; --dim) {
            const auto coord{remaining %
                             static_cast<std::size_t>(_local_size[dim])};
            remaining /= static_cast<std::size_t>(_local_size[dim]);
            offset += static_cast<std::ptrdiff_t>(coord + 1) * _strides[dim];
        }
        return offset;
    }
    /// Maps a flat index in [0, interior_size()) to an offset into
    /// local_grid().base() for the corresponding interior cell.
    auto interior_offset(std::size_t linear) const -> std::ptrdiff_t {
        std::ptrdiff_t offset{0};
        std::size_t remaining{linear};
        for (std::ptrdiff_t dim = NDims - 1; dim >= 0; --dim) {
            const auto extent{static_cast<std::size_t>(
                std::max(0, _local_size[dim] - 2))};
            if (extent == 0) {
                return 0;
            }
            const auto coord{remaining % extent};
            remaining /= extent;
            offset += static_cast<std::ptrdiff_t>(coord + 2) * _strides[dim];
        }
        return offset;
    }
    /// Number of shell cells (inner cells adjacent to the halo).
    auto shell_size() const -> std::size_t {
        return inner_size() - interior_size();
    }
    /// Maps a flat index in [0, shell_size()) to an offset into
    /// local_grid().base() for the corresponding shell cell.
    auto shell_offset(std::size_t linear) const -> std::ptrdiff_t {
        const auto l0{static_cast<std::size_t>(_local_size[0])};
        const auto l1{static_cast<std::size_t>(_local_size[1])};
        const auto l2{static_cast<std::size_t>(_local_size[2])};
        if (l0 < 3 || l1 < 3 || l2 < 3) {
            return inner_offset(linear);
        }

        auto offset_of{[&](std::size_t i, std::size_t j, std::size_t k) {
            return static_cast<std::ptrdiff_t>(i) * _strides[0] +
                   static_cast<std::ptrdiff_t>(j) * _strides[1] +
                   static_cast<std::ptrdiff_t>(k) * _strides[2];
        }};

        const auto b0{l1 * l2};
        const auto b1{b0};
        const auto b2{(l0 - 2) * l2};
        const auto b3{b2};
        const auto b4{(l0 - 2) * (l1 - 2)};

        auto remaining{linear};
        if (remaining < b0) {
            const auto j{remaining / l2 + 1};
            const auto k{remaining % l2 + 1};
            return offset_of(1, j, k);
        }
        remaining -= b0;
        if (remaining < b1) {
            const auto j{remaining / l2 + 1};
            const auto k{remaining % l2 + 1};
            return offset_of(l0, j, k);
        }
        remaining -= b1;
        if (remaining < b2) {
            const auto i{remaining / l2 + 2};
            const auto k{remaining % l2 + 1};
            return offset_of(i, 1, k);
        }
        remaining -= b2;
        if (remaining < b3) {
            const auto i{remaining / l2 + 2};
            const auto k{remaining % l2 + 1};
            return offset_of(i, l1, k);
        }
        remaining -= b3;
        if (remaining < b4) {
            const auto i{remaining / (l1 - 2) + 2};
            const auto j{remaining % (l1 - 2) + 2};
            return offset_of(i, j, 1);
        }
        remaining -= b4;
        const auto i{remaining / (l1 - 2) + 2};
        const auto j{remaining % (l1 - 2) + 2};
        return offset_of(i, j, l2);
    }
    auto fill(double value) {
        auto elements{inner_subarray().elements()};
        std::fill(elements.begin(), elements.end(), value);
    }
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
    auto world_size() const -> int { return _world_size; }
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
