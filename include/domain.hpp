#pragma once

#include <concepts>
#include <ranges>

template <typename T>
concept DomainConcept =
    std::ranges::range<typename T::Region> && requires(T domain) {
        domain.synchronize();
        { domain.elements() } -> std::ranges::range;
    };
