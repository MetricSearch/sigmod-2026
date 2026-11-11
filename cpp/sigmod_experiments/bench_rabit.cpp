#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>


#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/utils/space.hpp"

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

void bench_rabit(size_t dim) {
    std::vector<float> vector(dim);
    std::mt19937 gen(1413);
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
    }

    size_t bits = 2;                  // num of bits 
    std::vector<uint8_t> code(dim);  // code
    float f_add, f_rescale, f_error;
    auto config = rabitqlib::quant::faster_config(dim, bits);

    rabitqlib::quant::quantize_full_single(
        vector.data(),
        dim,
        bits,
        code.data(),
        f_add,
        f_rescale,
        f_error,
        rabitqlib::METRIC_IP,
        config
    );

    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = dist(gen);
    }

    auto ip_func = rabitqlib::simd::excode_ipimpl::ip64_fxu3_avx512;

    float c_1 = -0.5F;
    float k1xsumq = c_1 * std::accumulate(query.begin(), query.end(), 0.F);
    float g_add = 0.F;

    // Requires bits == 2, dim % 64 == 0, and an AVX2-capable CPU.
    // The current dim == 768 satisfies the dimension requirement.
    std::vector<uint8_t> packed_code(dim / 4);

    rabitqlib::simd::packing_2bit_excode_avx2(
        code.data(), packed_code.data(), dim
    );


    auto estimate_distance = [&] {
        return rabitqlib::quant::full_est_dist(
            packed_code.data(),
            query.data(),
            ip_func,
            dim,
            bits,
            f_add,
            f_rescale,
            g_add,
            k1xsumq
        );
    };

    const std::string label = "RaBitQ IP distance (dim=" + std::to_string(dim) +
                              ", bits=" + std::to_string(bits) + ", unpacked uint32)";
    ankerl::nanobench::Bench().epochs(50).epochIterations(1'000'000'0).run(label, [&] {
        ankerl::nanobench::doNotOptimizeAway(code.data());
        ankerl::nanobench::doNotOptimizeAway(query.data());
        ankerl::nanobench::doNotOptimizeAway(estimate_distance());
    });
}


// Adapted from docs/docs/rabitq/quantizer.md
int main() {
    for (int i = 1; i < 6; i++) {
        bench_rabit(512 * i);
    }
}
