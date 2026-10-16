#include <cstdint>
#include <random>
#include <vector>
#include <iostream>
#include <time.h>


#include "rabitqlib/quantization/rabitq.hpp"

int main() {
    size_t dim = 768;

    std::vector<float> vector(dim);
    std::vector<float> centroid(dim);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);

    // generate a random vector
    for (size_t i = 0; i < dim; ++i) {
        vector[i] = dist(gen);
        centroid[i] = dist(gen);
    }

    size_t bits = 2;                  // num of bits for total code
    std::vector<uint32_t> code(dim);  // code
    float f_add;                      // factors for estimating similarity
    float f_rescale;                  // factors for estimating similarity
    float f_error;                    // factors for computing error bounds

    clock_t tStart = clock();

    for (size_t i = 0; i < 10000; ++i) {
            // faster version, must init a config struct first
        rabitqlib::quant::quantize_full_single(
            vector.data(),
            centroid.data(),
            dim,
            bits,
            code.data(),
            f_add,
            f_rescale,
            f_error,
            rabitqlib::METRIC_L2
        );
    }

    size_t query_id = 585585;
    

    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}