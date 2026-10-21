#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

/// Invokes `f(std::integral_constant<std::size_t, I>{})` once for each index
/// `I` in the given `std::index_sequence`. This is the C++17-compatible
/// replacement for the C++20 pattern of an immediately-invoked lambda with an
/// explicit template parameter pack.
template <typename F, std::size_t... Is>
void for_each_index(F &&f, std::index_sequence<Is...>) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
}
