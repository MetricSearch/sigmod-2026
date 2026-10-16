// From sigmod_experiments:
// g++ -std=c++17 -O3 -mavx512f -mavx512vpopcntdq bench.cpp -o /tmp/b2sp_bench
//
// Run on a CPU supporting AVX-512F and AVX-512VPOPCNTDQ: /tmp/b2sp_bench
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "b2sp_avx512.hpp"
#include "nanobench.h"

// Return the indices of the largest half by absolute value, largest first.
std::vector<size_t> get_largest_coords(const std::vector<float>& vec) {
    // Copy vector
    std::vector<size_t> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by absolute value
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        const float abs_a = std::abs(vec[a]);
        const float abs_b = std::abs(vec[b]);
        return abs_a == abs_b ? a < b : abs_a > abs_b;
    });

    // Return
    indices.resize(vec.size() / 2);
    return indices;
}

void fill_evp(size_t blocks, const std::vector<float>& values, __m512i* pos, __m512i* neg) {
    // An 8x64=512 vector
    std::vector<uint64_t> p(blocks * 8), n(blocks * 8);

    // Gets the largest co-ords
    for (size_t i : get_largest_coords(values)) {
        // Sets the ith bit in the relevant p / n
        (values[i] >= 0 ? p : n)[i / 64] |= uint64_t{1} << (63 - i % 64);
    }

    for (size_t b = 0; b < blocks; ++b) {
        // Grabs 512 bits at the blocks * 8 position
        pos[b] = _mm512_loadu_si512(p.data() + b * 8);
        neg[b] = _mm512_loadu_si512(n.data() + b * 8);
    }
}

void fill_set_sign(size_t blocks, const std::vector<float>& values, __m512i* set, __m512i* sign) {
    std::vector<uint64_t> sets(blocks * 8), signs(blocks * 8);

    // Gets the largest co-ords
    for (size_t i : get_largest_coords(values)) {
        // Sets set bits
        sets[i / 64] |= uint64_t{1} << (63 - i % 64);

        // Set sign to 1 if this is a positive
        if (values[i] >= 0) {
            signs[i / 64] |= uint64_t{1} << (63 - i % 64);
        }
    }
    for (size_t b = 0; b < blocks; ++b) {
        set[b] = _mm512_loadu_si512(sets.data() + b * 8);
        sign[b] = _mm512_loadu_si512(signs.data() + b * 8);
    }
}

int bench_evp_at_dim(size_t dim) {
    const size_t blocks = (dim + 511) / 512;

    // Makes a random vector
    std::mt19937 rng(1413);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Make first random vector
    std::vector<float> vec1(dim);
    for (auto& value : vec1) {
        value = normal(rng);
    }

    // Make second random vector
    std::vector<float> vec2(dim);
    for (auto& value : vec2) {
        value = normal(rng);
    }

    std::vector<__m512i> a_pos(blocks), a_neg(blocks), b_pos(blocks), b_neg(blocks);
    fill_evp(blocks, vec1, a_pos.data(), a_neg.data());
    fill_evp(blocks, vec2, b_pos.data(), b_neg.data());

    const std::string label = "EVP b2sp AVX-512 (dim=" + std::to_string(dim) + ")";
    ankerl::nanobench::Bench().epochs(50).epochIterations(10'000'0).run(label, [&] {
        ankerl::nanobench::doNotOptimizeAway(a_pos);
        ankerl::nanobench::doNotOptimizeAway(a_neg);
        ankerl::nanobench::doNotOptimizeAway(b_pos);
        ankerl::nanobench::doNotOptimizeAway(b_neg);
        ankerl::nanobench::doNotOptimizeAway(
            b2sp(a_pos.data(), a_neg.data(), b_pos.data(), b_neg.data(), dim));
    });
    return 0;
}

int bench_evp_set_sign_at_dim(size_t dim) {
    const size_t blocks = (dim + 511) / 512;

    // Makes a random vector
    std::mt19937 rng(1413);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Make first random vector
    std::vector<float> vec1(dim);
    for (auto& value : vec1) {
        value = normal(rng);
    }

    // Make second random vector
    std::vector<float> vec2(dim);
    for (auto& value : vec2) {
        value = normal(rng);
    }

    std::vector<__m512i> a_set(blocks), a_sign(blocks), b_set(blocks), b_sign(blocks);
    fill_set_sign(blocks, vec1, a_set.data(), a_sign.data());
    fill_set_sign(blocks, vec2, b_set.data(), b_sign.data());

    const std::string label = "EVP b2sp set/sign AVX-512 (dim=" + std::to_string(dim) + ")";
    ankerl::nanobench::Bench().epochs(50).epochIterations(1'000'000'0).run(label, [&] {
        ankerl::nanobench::doNotOptimizeAway(a_set);
        ankerl::nanobench::doNotOptimizeAway(a_sign);
        ankerl::nanobench::doNotOptimizeAway(b_set);
        ankerl::nanobench::doNotOptimizeAway(b_sign);
        ankerl::nanobench::doNotOptimizeAway(
            set_sign_b2sp(a_set.data(), a_sign.data(), b_set.data(), b_sign.data(), dim));
    });
    return 0;
}


int main() {

    for (int i = 1; i < 6; i++) {
        bench_evp_at_dim(512 * i);
        bench_evp_set_sign_at_dim(512 * i);
    }
    
}
