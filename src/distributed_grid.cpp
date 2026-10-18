#include "distributed_grid.hpp"
#include "boundary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
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
        result += std::format("{}", value);
    }
    result += ']';
    return result;
}
} // namespace

auto DistributedStructuredGrid::init_topology(std::span<int, NDims> periods)
    -> void {
    /// Cartesian grid
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
    SPDLOG_DEBUG(std::format("World Rank {} -> Cart Rank {}; Coords: {}",
                             _cart_rank, _world_rank,
                             format_array(_cart_coords)));

    /// Find neighbors
    for (auto dim : std::views::iota(0, NDims)) {
        MPI_Cart_shift(_cart_comm, dim, 1, &_cart_neighbors[dim].first,
                       &_cart_neighbors[dim].second);
        SPDLOG_DEBUG(
            "Rank {}: My neighbors in Dimension {} are Rank {} and Rank {}",
            _cart_rank, dim, _cart_neighbors[dim].first,
            _cart_neighbors[dim].second);
    }

    // Initialize local grid
    for (auto dim : std::views::iota(0, NDims)) {
        auto unit_size{std::floor(_global_size[dim] / _cart_dims[dim])};
        _local_start[dim] = _cart_coords[dim] * unit_size;
        _local_size[dim] = (_cart_coords[dim] < _cart_dims[dim] - 1)
                               ? unit_size
                               : _global_size[dim] - _local_start[dim];
        _exterior_size[dim] = _local_size[dim] + 2;
    }
    SPDLOG_DEBUG(
        std::format("Rank {}: My partition starts at {} and has size {}",
                    _cart_rank, format_array(_local_start),
                    format_array(_local_size)));
}

auto DistributedStructuredGrid::init_transfer_types() -> void {
    MPI_Datatype element_type;
    MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(double), &element_type);
    for (auto dim : std::views::iota(0, NDims)) {
        std::array<int, NDims> interior_face_subsizes;
        std::array<int, NDims> exterior_face_subsizes;
        std::pair<std::array<int, NDims>, std::array<int, NDims>>
            interior_face_starts;
        std::pair<std::array<int, NDims>, std::array<int, NDims>>
            exterior_face_starts;
        for (auto odim : std::views::iota(0, NDims)) {
            // NOTE: May be improved with a disjoint fill and assignment at
            // [dim]
            if (dim == odim) {
                exterior_face_starts.first[odim] = 0;
                interior_face_starts.first[odim] = 1;
                exterior_face_starts.second[odim] = _exterior_size[odim] - 1;
                interior_face_starts.second[odim] = _local_size[odim];
                exterior_face_subsizes[odim] = 1;
                interior_face_subsizes[odim] = 1;
            } else {
                // NOTE: Only interior cells are exchanged, corners are excluded
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
    for (auto dim : std::views::iota(0, NDims)) {
        MPI_Type_free(&_interior_face_types[dim].first);
        MPI_Type_free(&_exterior_face_types[dim].first);
        MPI_Type_free(&_interior_face_types[dim].second);
        MPI_Type_free(&_exterior_face_types[dim].second);
    }
}

auto DistributedStructuredGrid::init_requests() -> void {
    for (auto dim : std::views::iota(0, NDims)) {
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

template <std::size_t... I>
auto DistributedStructuredGrid::create_subarray(std::array<int, NDims> from,
                                                std::array<int, NDims> to,
                                                std::index_sequence<I...>)
    -> Subarray {
    return _mdarray(boost::multi::index_range{from[I], to[I]}...);
}

auto DistributedStructuredGrid::inner_subarray() -> Subarray {
    std::array<int, NDims> inner_from, inner_to;
    for (auto dim : std::views::iota(0, NDims)) {
        inner_from[dim] = 1;
        inner_to[dim] = _local_size[dim] + 1;
    }
    return create_subarray(inner_from, inner_to,
                           std::make_index_sequence<NDims>{});
}

auto DistributedStructuredGrid::get_face(std::ptrdiff_t axis, FaceEnd end,
                                         FaceKind kind) -> Subarray {
    std::array<int, NDims> from;
    std::array<int, NDims> to;
    if (kind == FaceKind::Interior) {
        if (end == FaceEnd::First) {
            for (auto dim : std::views::iota(0, NDims)) {
                from[dim] = 1;
                to[dim] = (dim == axis) ? 2 : _exterior_size[dim] - 1;
            }
        } else {
            for (auto dim : std::views::iota(0, NDims)) {
                to[dim] = _exterior_size[dim] - 1;
                from[dim] = (dim == axis) ? _exterior_size[dim] - 2 : 1;
            }
        }
    } else {

        if (end == FaceEnd::First) {
            for (auto dim : std::views::iota(0, NDims)) {
                from[dim] = (dim == axis) ? 0 : 1;
                to[dim] = (dim == axis) ? 1 : _exterior_size[dim] - 1;
            }
        } else {
            for (auto dim : std::views::iota(0, NDims)) {
                to[dim] = (dim == axis) ? _exterior_size[dim]
                                        : _exterior_size[dim] - 1;
                from[dim] = (dim == axis) ? _exterior_size[dim] - 1 : 1;
            }
        }
    }
    return create_subarray(from, to, std::make_index_sequence<NDims>{});
}

auto DistributedStructuredGrid::set_local_boundaries(
    std::span<const AxisBoundary, NDims> axis_boundaries) -> void {
    _neumann_boundary_count = 0;

    auto apply_boundary{[&](int axis, FaceEnd end) {
        auto interior_face{get_face(axis, end, FaceKind::Interior)};
        auto exterior_face{get_face(axis, end, FaceKind::Exterior)};
        const auto &faces{std::get<AxisFaces>(axis_boundaries[axis])};
        const NonPeriodicBoundary &boundary{end == FaceEnd::First ? faces.first
                                                                  : faces.last};

        if (const auto *dirichlet = std::get_if<DirichletBoundary>(&boundary)) {
            DirichletApplicator{dirichlet->value, std::move(exterior_face)}
                .apply();
        } else {
            const auto *neumann{std::get_if<NeumannBoundary>(&boundary)};
            assert(neumann != nullptr);
            _neumann_boundaries[_neumann_boundary_count++]
                = {neumann->delta,
                   std::move(exterior_face),
                   std::move(interior_face)};
        }
    }};

    for (auto dim : std::views::iota(0, NDims)) {
        if (_cart_neighbors[dim].first == MPI_PROC_NULL)
            apply_boundary(dim, FaceEnd::First);
        if (_cart_neighbors[dim].second == MPI_PROC_NULL)
            apply_boundary(dim, FaceEnd::Last);
    }
}

DistributedStructuredGrid::DistributedStructuredGrid(
    std::span<const int, NDims> size, BoundaryConditions boundary_specification,
    double cell_size, double default_value)
    : _cell_size(cell_size) {
    MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_world_size);
    std::ranges::copy(size, _global_size.begin());
    SPDLOG_DEBUG("Process {}/{} (rank {})", _world_rank + 1, _world_size,
                 _world_rank);

    std::array<int, NDims> periods{};
    for (auto dim : std::views::iota(0, NDims)) {
        periods[dim] =
            std::get_if<PeriodicBoundary>(&boundary_specification[dim]) !=
            nullptr;
    }
    init_topology(periods);

    std::array<std::ptrdiff_t, NDims> extensions;
    for (auto dim : std::views::iota(0, NDims)) {
        extensions[dim] = _exterior_size[dim];
    }
    auto extents{std::apply(
        [](auto... s) {
            return boost::multi::extents_t<NDims>(
                static_cast<std::ptrdiff_t>(s)...);
        },
        extensions)};
    _mdarray = boost::multi::array<double, NDims>(extents, default_value);

    init_transfer_types();
    set_local_boundaries(boundary_specification);
    init_requests();
}

DistributedStructuredGrid::~DistributedStructuredGrid() {
    deinit_transfer_types();
    deinit_requests();
    MPI_Comm_free(&_cart_comm);
}

auto DistributedStructuredGrid::synchronize() -> void {
    MPI_Startall(_requests.size(), _requests.data());
    for (auto i = 0u; i < _neumann_boundary_count; ++i) {
        _neumann_boundaries[i].apply();
    }
    std::array<MPI_Status, _requests.size()> statuses;
    MPI_Waitall(_requests.size(), _requests.data(), statuses.data());
}
