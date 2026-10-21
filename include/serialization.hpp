#pragma once

#include <fstream>
#include <multi/array_ref.hpp>
#include <string>

struct PPM {
    static constexpr std::string extension{"ppm"};
    unsigned max_value;
};

inline auto serialize(std::ostream &file, const boost::multi::array<double, 2> &array,
                const PPM &ppm) {
    auto [is, js] = array.extents();
    file << "P3\n" << is.size() << " " << js.size() << "\n" << ppm.max_value
         << "\n";
    for (auto j : js) { // WARN: Inefficient iteration order
        for (auto i : is) {
            auto value{static_cast<unsigned>(array[j][i])};
            file << value << " " << value << " " << value << " ";
        }
        file << "\n";
    }

    return true;
}
