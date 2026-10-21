#pragma once

#include <cstddef>
#include <pugixml.hpp>

#include <charconv>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline auto required_child(const pugi::xml_node &node,
                           std::string_view child_name) -> pugi::xml_node {
    if (auto child{node.child(child_name.data())}; child) {
        return child;
    }
    throw ConfigError{"Missing required element <" + std::string(child_name) +
                       "> in <" + node.name() + ">"};
}

template <typename T>
auto parse_value(std::string_view text) -> T {
    T value{};
    auto [ptr,
          ec]{std::from_chars(text.data(), text.data() + text.size(), value)};
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        throw ConfigError{"Expected a number, got \"" + std::string(text) +
                           "\""};
    }
    return value;
}

inline auto child_attribute_text(const pugi::xml_node &node,
                                 std::string_view child_name,
                                 std::string_view attribute)
    -> std::string_view {
    return required_child(node, child_name).attribute(attribute).as_string();
}

template <typename T>
auto child_attribute_value(const pugi::xml_node &node,
                           std::string_view child_name,
                           std::string_view attribute) -> T {
    return parse_value<T>(child_attribute_text(node, child_name, attribute));
}

inline auto child_text(const pugi::xml_node &node, std::string_view child_name)
    -> std::string_view {
    return required_child(node, child_name).child_value();
}

template <typename T>
auto child_value(const pugi::xml_node &node, std::string_view child_name) -> T {
    return parse_value<T>(child_text(node, child_name));
}

inline auto child_pair(const pugi::xml_node &node, std::string_view child_name)
    -> std::pair<double, double> {
    auto child{required_child(node, child_name)};
    std::string_view text{child.child_value()};
    double values[2]{};
    std::size_t value_count{0};
    std::string_view remaining{text};
    while (!remaining.empty()) {
        auto pos{remaining.find(' ')};
        auto token{remaining.substr(0, pos)};
        if (!token.empty()) {
            if (value_count >= 2) {
                throw ConfigError{"Failed to parse <" +
                                  std::string(child_name) + "> in <" +
                                  node.name() +
                                  ">: expected two numbers, got \"" +
                                  std::string(text) + "\""};
            }
            values[value_count++] = parse_value<double>(token);
        }
        if (pos == std::string_view::npos) {
            remaining = {};
        } else {
            remaining.remove_prefix(pos + 1);
        }
    }
    if (value_count != 2) {
        throw ConfigError{"Failed to parse <" + std::string(child_name) +
                          "> in <" + node.name() +
                          ">: expected two numbers, got \"" +
                          std::string(text) + "\""};
    }
    return {values[0], values[1]};
}
