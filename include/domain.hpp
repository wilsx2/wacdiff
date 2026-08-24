#pragma once

#include <concepts>
#include <ranges>

template <typename T>
concept DomainConcept = std::ranges::range<typename T::Region> &&
                        requires(T domain, std::ptrdiff_t axis, bool first,
                                 bool interior) {
    domain.synchronize();
    { domain.elements() } -> std::ranges::range;
    { domain.inner() } -> std::ranges::range;
    { domain.local_face(axis, first, interior) } -> std::ranges::range;
};
