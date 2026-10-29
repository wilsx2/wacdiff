#pragma once

#include <cstdint>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <mpi.h>
#include <multi/array.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

enum class FaceEnd  { First, Last };
enum class FaceKind { Interior, Exterior };

class DistributedStructuredGrid {
public:
    static constexpr std::ptrdiff_t NDims = 3;

    using Subarray = decltype(std::declval<boost::multi::array<double, NDims> &>()(
        boost::multi::index_range{}, boost::multi::index_range{},
        boost::multi::index_range{}));

private:
    // Data and Partitioning
    std::array<int32_t, NDims> _global_size;
    std::array<int32_t, NDims> _local_size;
    std::array<int32_t, NDims> _local_start;
    std::array<int32_t, NDims> _exterior_size;
    boost::multi::array<double, NDims> _mdarray;

    // Metadata
    double _cell_size;

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
    auto init_topology() -> void;
    auto init_transfer_types() -> void;
    auto deinit_transfer_types() -> void;
    auto init_requests() -> void;
    auto deinit_requests() -> void;
    template <std::size_t... I>
    auto create_subarray(std::array<int, NDims> from, std::array<int, NDims> to,
                         std::index_sequence<I...>) -> Subarray {
        return _mdarray(boost::multi::index_range{from[I], to[I]}...);
    }
    auto get_face(std::ptrdiff_t axis, FaceEnd end, FaceKind kind) -> Subarray;

    template <typename F>
    static auto invoke_summand(F &f, std::ptrdiff_t i, std::ptrdiff_t j,
                               std::ptrdiff_t k) -> double {
        using Result = std::invoke_result_t<F &, std::ptrdiff_t,
                                            std::ptrdiff_t, std::ptrdiff_t>;
        if constexpr (std::is_void_v<Result>) {
            f(i, j, k);
            return 0.0;
        } else {
            static_assert(std::is_arithmetic_v<Result>);
            return f(i, j, k);
        }
    }

public:
    DistributedStructuredGrid(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid &operator=(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid &operator=(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid(
        const std::array<int, NDims> &size,
        double block_size = 1);
    ~DistributedStructuredGrid();
    auto begin_halo_exchange() -> void;
    auto await_halo_exchange() -> void;
    auto inner_subarray() -> Subarray {
        const auto &dims = _exterior_size;
        return create_subarray({1, 1, 1},
                               {dims[0] - 1, dims[1] - 1, dims[2] - 1},
                               std::make_index_sequence<NDims>{});
    }
    auto shell_subarray() -> Subarray {
        const auto &dims = _exterior_size;
        return create_subarray({2, 2, 2},
                               {dims[0] - 2, dims[1] - 2, dims[2] - 2},
                               std::make_index_sequence<NDims>{});
    }
    auto local_grid() -> boost::multi::array<double, NDims> & { return _mdarray; }
    auto local_grid() const -> const boost::multi::array<double, NDims> & {
        return _mdarray;
    }
    auto local_size() const -> const std::array<int, NDims> & {
        return _local_size;
    }
    auto exterior_size() const -> const std::array<int, NDims> & {
        return _exterior_size;
    }
    auto local_start() const -> const std::array<int, NDims> & {
        return _local_start;
    }
    auto global_size() const -> const std::array<int, NDims> & {
        return _global_size;
    }
    auto cell_size() const -> double { return _cell_size; }

    template <typename F>
    auto for_all_in_shell(F&& f) -> double {
        const auto &dims = _exterior_size;
        auto sum = 0.0;

        #pragma omp for schedule(static) collapse(2) nowait
        for (std::ptrdiff_t j = 1; j < dims[1] - 1; j++) {
            for (std::ptrdiff_t k = 1; k < dims[2] - 1; k++) {
                sum += invoke_summand(f, 1, j, k);
                sum += invoke_summand(f, dims[0] - 2, j, k);
            }
        }
        #pragma omp for schedule(static) collapse(2) nowait
        for (std::ptrdiff_t i = 2; i < dims[0] - 2; i++) {
            for (std::ptrdiff_t k = 1; k < dims[2] - 1; k++) {
                sum += invoke_summand(f, i, 1, k);
                sum += invoke_summand(f, i, dims[1] - 2, k);
            }
        }
        #pragma omp for schedule(static) collapse(2) nowait
        for (std::ptrdiff_t i = 2; i < dims[0] - 2; i++) {
            for (std::ptrdiff_t j = 2; j < dims[1] - 2; j++) {
                sum += invoke_summand(f, i, j, 1);
                sum += invoke_summand(f, i, j, dims[2] - 2);
            }
        }
        return sum;
    }

    template <typename F>
    auto for_all_on_shell(F&& f) -> double {
        const auto &dims = _exterior_size;
        auto sum = 0.0;

        #pragma omp for schedule(static) collapse(2) nowait
        for (std::ptrdiff_t i = 2; i < dims[0] - 2; i++) {
            for (std::ptrdiff_t j = 2; j < dims[1] - 2; j++) {
                #pragma omp simd reduction(+ : sum)
                for (std::ptrdiff_t k = 2; k < dims[2] - 2; k++) {
                    sum += invoke_summand(f, i, j, k);
                }
            }
        }
        return sum;
    }

    auto comm() const -> MPI_Comm { return _cart_comm; }
    auto world_rank() const -> int { return _world_rank; }
    auto world_size() const -> int { return _world_size; }
};
