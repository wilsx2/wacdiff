#include "distributed_grid.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <variant>

namespace {
template <typename Array>
auto format_array(const Array &values) -> std::string {
    std::string result{"["};
    bool first{true};
    for (const auto &value : values) {
        if (!first) {
            result += ", ";
        }
        first = false;
        result += std::to_string(value);
    }
    result += ']';
    return result;
}
} // namespace

auto DistributedStructuredGrid::init_topology()
    -> void {
    /// Cartesian grid
    std::array<int, NDims> periods{true, true, true};
    std::fill(_cart_dims.begin(), _cart_dims.end(), 0);
    MPI_Dims_create(_world_size, NDims, _cart_dims.data());
    MPI_Cart_create(MPI_COMM_WORLD, NDims, _cart_dims.data(), periods.data(), 0,
                    &_cart_comm);

    if (_cart_comm == MPI_COMM_NULL) {
        SPDLOG_CRITICAL("Process {} not included in Cartesian communicator",
                        _world_rank);
        std::exit(EXIT_FAILURE);
    }
    /// Find coordinates
    MPI_Cart_coords(_cart_comm, _world_rank, NDims, _cart_coords.data());
    MPI_Cart_rank(_cart_comm, _cart_coords.data(), &_cart_rank);
    SPDLOG_DEBUG("World Rank " + std::to_string(_cart_rank) +
                 " -> Cart Rank " + std::to_string(_world_rank) +
                 "; Coords: " + format_array(_cart_coords));

    /// Find neighbors
    for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
        MPI_Cart_shift(_cart_comm, dim, 1, &_cart_neighbors[dim].first,
                       &_cart_neighbors[dim].second);
        SPDLOG_DEBUG(
            "Rank {}: My neighbors in Dimension {} are Rank {} and Rank {}",
            _cart_rank, dim, _cart_neighbors[dim].first,
            _cart_neighbors[dim].second);
    }

    for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
        _partition_size[dim] =
            (_global_size[dim] + _cart_dims[dim] - 1) / _cart_dims[dim];
        _local_start[dim] = _cart_coords[dim] * _partition_size[dim];
        _local_size[dim] = std::min(_partition_size[dim],
                                    _global_size[dim] - _local_start[dim]);
        _exterior_size[dim] = _local_size[dim] + 2;

        if (_local_size[dim] < 1) {
            SPDLOG_CRITICAL(
                "Rank {}: {} cells along axis {} cannot be split over {} "
                "ranks without leaving a rank empty",
                _cart_rank, _global_size[dim], dim, _cart_dims[dim]);
            std::exit(EXIT_FAILURE);
        }
    }
    SPDLOG_DEBUG("Rank " + std::to_string(_cart_rank) +
                 ": My partition starts at " + format_array(_local_start) +
                 " and has size " + format_array(_local_size));
}

auto DistributedStructuredGrid::init_transfer_types() -> void {
    MPI_Datatype element_type;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(double), &element_type);
    for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
        std::array<int, NDims> interior_face_subsizes;
        std::array<int, NDims> exterior_face_subsizes;
        std::pair<std::array<int, NDims>, std::array<int, NDims>>
            interior_face_starts;
        std::pair<std::array<int, NDims>, std::array<int, NDims>>
            exterior_face_starts;
        for (std::ptrdiff_t odim = 0; odim < NDims; ++odim) {
            if (dim == odim) {
                exterior_face_starts.first[odim] = 0;
                interior_face_starts.first[odim] = 1;
                exterior_face_starts.second[odim] = _exterior_size[odim] - 1;
                interior_face_starts.second[odim] = _local_size[odim];
                exterior_face_subsizes[odim] = 1;
                interior_face_subsizes[odim] = 1;
            } else {
                exterior_face_starts.first[odim] = 1;
                interior_face_starts.first[odim] = 1;
                exterior_face_starts.second[odim] = 1;
                interior_face_starts.second[odim] = 1;
                exterior_face_subsizes[odim] = _local_size[odim];
                interior_face_subsizes[odim] = _local_size[odim];
            }
        }

        MPI_Type_create_subarray(
            NDims, _exterior_size.data(), interior_face_subsizes.data(),
            interior_face_starts.first.data(), MPI_ORDER_C, element_type,
            &_interior_face_types[dim].first);
        MPI_Type_create_subarray(
            NDims, _exterior_size.data(), interior_face_subsizes.data(),
            interior_face_starts.second.data(), MPI_ORDER_C, element_type,
            &_interior_face_types[dim].second);
        MPI_Type_create_subarray(
            NDims, _exterior_size.data(), exterior_face_subsizes.data(),
            exterior_face_starts.first.data(), MPI_ORDER_C, element_type,
            &_exterior_face_types[dim].first);
        MPI_Type_create_subarray(
            NDims, _exterior_size.data(), exterior_face_subsizes.data(),
            exterior_face_starts.second.data(), MPI_ORDER_C, element_type,
            &_exterior_face_types[dim].second);

        MPI_Type_commit(&_interior_face_types[dim].first);
        MPI_Type_commit(&_exterior_face_types[dim].first);
        MPI_Type_commit(&_interior_face_types[dim].second);
        MPI_Type_commit(&_exterior_face_types[dim].second);
    }
}

auto DistributedStructuredGrid::deinit_transfer_types() -> void {
    for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
        MPI_Type_free(&_interior_face_types[dim].first);
        MPI_Type_free(&_exterior_face_types[dim].first);
        MPI_Type_free(&_interior_face_types[dim].second);
        MPI_Type_free(&_exterior_face_types[dim].second);
    }
}

auto DistributedStructuredGrid::init_requests() -> void {
    _requests_in_flight = false;

    for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
        auto neighbors{_cart_neighbors[dim]};

        MPI_Send_init(_mdarray.base(), 1, _interior_face_types[dim].first,
                      neighbors.first, 0, _cart_comm,
                      _requests.data() + dim * 4 + 0);
        MPI_Recv_init(_mdarray.base(), 1, _exterior_face_types[dim].first,
                      neighbors.first, 1, _cart_comm,
                      _requests.data() + dim * 4 + 1);
        MPI_Recv_init(_mdarray.base(), 1, _exterior_face_types[dim].second,
                      neighbors.second, 0, _cart_comm,
                      _requests.data() + dim * 4 + 2);
        MPI_Send_init(_mdarray.base(), 1, _interior_face_types[dim].second,
                      neighbors.second, 1, _cart_comm,
                      _requests.data() + dim * 4 + 3);
    }
}

auto DistributedStructuredGrid::deinit_requests() -> void {
    for (auto &request : _requests) {
        MPI_Request_free(&request);
    }
}

auto DistributedStructuredGrid::get_face(std::ptrdiff_t axis, FaceEnd end,
                                         FaceKind kind) -> Subarray {
    std::array<int, NDims> from;
    std::array<int, NDims> to;
    if (kind == FaceKind::Interior) {
        if (end == FaceEnd::First) {
            for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
                from[dim] = 1;
                to[dim] = (dim == axis) ? 2 : _exterior_size[dim] - 1;
            }
        } else {
            for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
                to[dim] = _exterior_size[dim] - 1;
                from[dim] = (dim == axis) ? _exterior_size[dim] - 2 : 1;
            }
        }
    } else {
        if (end == FaceEnd::First) {
            for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
                from[dim] = (dim == axis) ? 0 : 1;
                to[dim] = (dim == axis) ? 1 : _exterior_size[dim] - 1;
            }
        } else {
            for (std::ptrdiff_t dim = 0; dim < NDims; ++dim) {
                to[dim] = (dim == axis) ? _exterior_size[dim]
                                        : _exterior_size[dim] - 1;
                from[dim] = (dim == axis) ? _exterior_size[dim] - 1 : 1;
            }
        }
    }
    return create_subarray(from, to, std::make_index_sequence<NDims>{});
}

DistributedStructuredGrid::DistributedStructuredGrid(
    const std::array<int, NDims> &size, double cell_size)
    : _cell_size(cell_size) {
    MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_world_size);
    std::copy(size.begin(), size.end(), _global_size.begin());
    SPDLOG_DEBUG("Process {}/{} (rank {})", _world_rank + 1, _world_size,
                 _world_rank);

    init_topology();

    auto extents = boost::multi::extents_t<NDims>(
        _exterior_size[0],
        _exterior_size[1],
        _exterior_size[2]);
    // NOTE: Uninitialized
    _mdarray = boost::multi::array<double, NDims>(extents);

    // Keep transparent huge pages from collapsing the per-thread first-touch
    // regions into one page the kernel may then migrate across NUMA domains.
    const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto base = reinterpret_cast<std::uintptr_t>(_mdarray.base());
    const auto offset = base % page_size;
    const auto length =
        static_cast<std::size_t>(_mdarray.num_elements()) * sizeof(double) +
        offset;
    if (madvise(reinterpret_cast<void *>(base - offset), length,
                MADV_NOHUGEPAGE) != 0) {
        SPDLOG_DEBUG("madvise(MADV_NOHUGEPAGE) failed: {}",
                     std::strerror(errno));
    }

    init_transfer_types();
    init_requests();
}

DistributedStructuredGrid::~DistributedStructuredGrid() {
    deinit_transfer_types();
    deinit_requests();
    MPI_Comm_free(&_cart_comm);
}

auto DistributedStructuredGrid::begin_halo_exchange() -> void {
    if (!_requests_in_flight) {
        MPI_Startall(_requests.size(), _requests.data());
        _requests_in_flight = true;
    }
}

auto DistributedStructuredGrid::await_halo_exchange() -> void {
    if (_requests_in_flight) {
        std::array<MPI_Status, NDims * 4> statuses;
        MPI_Waitall(_requests.size(), _requests.data(), statuses.data());
        _requests_in_flight = false;
    }
}
