#include <iostream>
#include <cstdint>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

const float conv[2] = {-1.0f, 1.0f};


float dist(float* query, uint64_t* set_bits, uint64_t* signs, const int_fast32_t dim) {
    /**
     * Computes EVP* over a dense f32 query.
     * The quantised data is represented by two packed bitstrings:
     *  * set_bits is set to 0 if the EVP vertex co-ordinate is 0, and is 1 if it is either 1 / -1
     *  * The ith 1 in the set_bits vector corresponds to the ith element of the signs vector. set_bits is 0 if the EVP vertex is -1, or 1 if 1.
     *  * We assume the {d/2, d} EVP is being used.
     * dim is the dimension of query
     */
    int_fast32_t sign_ptr = 0;

    // Gets everything (+1s and -1s)
    float total_acc = 0.0;
    const int total_loops = (dim + 63) / 64;

    int lsb, conv_ptr, query_ptr;
    uint64_t current_set_bits;

    for (int i = 0; i < total_loops; ++i) {
        current_set_bits = set_bits[i];

        while (current_set_bits) {
            // Where is the least significant set bit
            lsb = __builtin_ffsll(current_set_bits);

            // Zero out that bit
            current_set_bits &= current_set_bits - 1;

            // Gets the positive / negative index. Returns 1 if pos, 0 if neg.
            conv_ptr = (signs[sign_ptr / 64] >> (sign_ptr % 64)) & 1;

            // Find the matching co-ordinate of the query
            query_ptr = (i << 6) + lsb - 1;
            
            // Adds the sign multiplied query to the vector
            total_acc += conv[conv_ptr] * query[query_ptr];

            // Get the next sign
            ++sign_ptr;
        }
    }
    
    return total_acc;
}

// Build an optimized benchmark with:
// c++ -std=c++17 -O3 src/engine/evp_star.cpp -o /tmp/evp_star_bench
int main()
{
    constexpr int_fast32_t dim = 384;
    constexpr std::size_t item_count = 1'000'000;
    constexpr std::size_t set_words = dim / 64;
    constexpr std::size_t sign_words = (dim / 2) / 64;
    constexpr uint64_t seed = 12345;
    using PackedItem = std::array<uint64_t, set_words + sign_words>;
    using Clock = std::chrono::steady_clock;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> query_value(-1.0f, 1.0f);
    std::array<float, dim> query;
    for (float& value : query) value = query_value(rng);

    std::cout << "Generating " << item_count << " unique packed items, dim="
              << dim << ", seed=" << seed << "..." << std::endl;
    const auto generation_start = Clock::now();
    std::vector<PackedItem> items;
    items.reserve(item_count);
    std::array<int, dim> positions;
    std::iota(positions.begin(), positions.end(), 0);

    while (items.size() < item_count) {
        while (items.size() < item_count) {
            PackedItem item{};
            std::shuffle(positions.begin(), positions.end(), rng);
            // Exactly dim/2 randomly chosen coordinates are nonzero.
            for (int i = 0; i < dim / 2; ++i) {
                const int position = positions[i];
                item[position / 64] |= uint64_t{1} << (position % 64);
            }
            // Each sign bit independently selects +1 or -1, in coordinate order.
            for (std::size_t word = 0; word < sign_words; ++word) {
                item[set_words + word] = rng();
            }
            items.push_back(item);
        }
        // Compare all packed words to guarantee uniqueness; refill if needed.
        std::sort(items.begin(), items.end());
        items.erase(std::unique(items.begin(), items.end()), items.end());
    }
    // Restore random traversal order after checking uniqueness.
    std::shuffle(items.begin(), items.end(), rng);
    const auto generation_end = Clock::now();

    // Allocation, generation, deduplication and printing are outside the timer.
    std::vector<float> results(item_count);
    const auto start = Clock::now();
    for (std::size_t i = 0; i < item_count; ++i) {
        results[i] = dist(query.data(), items[i].data(),
                          items[i].data() + set_words, dim);
    }
    const auto end = Clock::now();

    // Consume every result outside the timed loop to keep the work observable.
    const double checksum = std::accumulate(results.begin(), results.end(), 0.0);
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double generation_seconds =
        std::chrono::duration<double>(generation_end - generation_start).count();
    std::cout << std::fixed << std::setprecision(3)
              << "Generation and uniqueness check: " << generation_seconds << " s\n"
              << "Query scan (" << item_count << " items): " << seconds * 1000.0 << " ms\n"
              << "Time per item: " << seconds * 1e9 / item_count << " ns\n"
              << "Throughput: " << item_count / seconds / 1e6 << " million items/s\n"
              << std::setprecision(9) << "Checksum: " << checksum << '\n';
    return 0;
}
