#ifndef EVP_QUANTIZATION_HPP
#define EVP_QUANTIZATION_HPP

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>

namespace EVP {

    /**
     * @brief Calculates the median of a vector of doubles.
     * @param row The input vector.
     * @return The median value.
     */
    inline double calculate_median(const std::vector<double>& row) {
        if (row.empty()) {
            return 0.0;
        }
        
        std::vector<double> sorted_row = row;
        size_t n = sorted_row.size();
        
        // first sort the array
        std::sort(sorted_row.begin(), sorted_row.end());

        // Check if the number is even or odd
        if (n % 2 != 0) {
            // If number of element is odd, return the middle
            // element
            return sorted_row[n / 2];
        }
        else {
            // If number of element is even, return the average
            // of the two middle elements
            return (sorted_row[(n - 1) / 2] + sorted_row[n / 2]) / 2.0;
        }
    }

    /**
     * @brief Ternary row mask based on median threshold.
     * Maps to different nodes later on Mac.
     * @param row Input row vector.
     * @return Vector of double (or int8_t cast) with values -1, 0, or 1.
     */
    inline std::vector<double> get_evp_ternary_row_mask(const std::vector<double>& row) {
        size_t row_len = row.size();
        if (row_len == 0) return {};

        size_t x = (row_len + 1) / 2; // Equivalent to int(np.ceil(row_len / 2))
        double median_row = calculate_median(row);

        std::vector<double> n_row(row_len, 0.0);

        size_t accounted = 0;
        for (size_t i = 0; i < row_len; ++i) {
            if (row[i] > median_row) {
                n_row[i] = 1.0;
                accounted++;
            } else if (row[i] < -median_row) {
                n_row[i] = -1.0;
                accounted++;
            }
        }

        if (accounted < x) {
            std::vector<size_t> zero_indices;
            for (size_t i = 0; i < row_len; ++i) {
                if (n_row[i] == 0) {
                    zero_indices.push_back(i);
                }
            }
            
            size_t needed = x - accounted;
            size_t count = 0;
            for (size_t i : zero_indices) {
                if (count < needed) {
                    n_row[i] = 1.0;
                    count++;
                } else {
                    break;
                }
            }
        }

        return n_row;
    }

    /**
     * @brief EVP in -1, 0, 1 form (Richards style).
     * @param data Matrix (vector of rows).
     * @return Vector of vectors of int8_t.
     */
    inline std::vector<std::vector<int8_t>> get_evp_richard(const std::vector<std::vector<double>>& data) {
        std::vector<std::vector<int8_t>> d_res;
        d_res.reserve(data.size());

        for (const auto& row : data) {
            double median_row = calculate_median(row);
            std::vector<int8_t> d_plus_row;
            std::vector<int8_t> d_minus_row;
            
            d_plus_row.reserve(row.size());
            d_minus_row.reserve(row.size());

            for (double val : row) {
                d_plus_row.push_back(val > median_row ? 1 : 0);
                d_minus_row.push_back(val < -median_row ? -1 : 0);
            }

            // Combine: d_plus + d_minus
            std::vector<int8_t> combined_row;
            combined_row.reserve(row.size());
            for (size_t i = 0; i < row.size(); ++i) {
                combined_row.push_back(d_plus_row[i] + d_minus_row[i]);
            }

            d_res.push_back(std::move(combined_row));
        }

        return d_res;
    }

    /**
     * @brief EVP ternary form using row mask.
     * @param data Matrix (vector of rows).
     * @return Vector of vectors of double.
     */
    inline std::vector<std::vector<double>> get_evp_ternary(const std::vector<std::vector<double>>& data) {
        std::vector<std::vector<double>> d_res;
        d_res.reserve(data.size());

        for (const auto& row : data) {
            d_res.push_back(get_evp_ternary_row_mask(row));
        }

        return d_res;
    }

    /**
     * @brief Binary row split.
     * @param row Input row vector.
     * @param d_minus Output vector for negative values (> 0 if row < -median).
     * @param d_plus Output vector for positive values (> 0 if row > median).
     */
    inline void get_evp_binary_row(const std::vector<double>& row, 
                                   std::vector<uint8_t>& d_minus, 
                                   std::vector<uint8_t>& d_plus) {
        double median_row = calculate_median(row);

        d_minus.resize(row.size());
        d_plus.resize(row.size());

        for (size_t i = 0; i < row.size(); ++i) {
            d_plus[i] = (row[i] > median_row) ? 1 : 0;
            d_minus[i] = (row[i] < -median_row) ? 1 : 0;
        }
    }

    /**
     * @brief EVP binary form (v+ v- concatenated).
     * @param data Matrix (vector of rows).
     * @return Concatenated matrix where each row is [negative_mask | positive_mask].
     */
    inline std::vector<std::vector<uint8_t>> get_evp_binary(const std::vector<std::vector<double>>& data) {
        std::vector<std::vector<uint8_t>> d_minus;
        std::vector<std::vector<uint8_t>> d_plus;
        
        d_minus.reserve(data.size());
        d_plus.reserve(data.size());

        for (const auto& row : data) {
            std::vector<uint8_t> d_minus_row;
            std::vector<uint8_t> d_plus_row;
            
            get_evp_binary_row(row, d_minus_row, d_plus_row);
            
            d_minus.push_back(std::move(d_minus_row));
            d_plus.push_back(std::move(d_plus_row));
        }

        // Concatenate horizontally: [d_minus | d_plus]
        std::vector<std::vector<uint8_t>> evp_array;
        evp_array.reserve(data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            std::vector<uint8_t> combined_row;
            combined_row.reserve(d_minus[i].size() + d_plus[i].size());
            
            combined_row.insert(combined_row.end(), d_minus[i].begin(), d_minus[i].end());
            combined_row.insert(combined_row.end(), d_plus[i].begin(), d_plus[i].end());
            
            evp_array.push_back(std::move(combined_row));
        }

        return evp_array;
    }

} // namespace EVP

#endif // EVP_QUANTIZATION_HPP
