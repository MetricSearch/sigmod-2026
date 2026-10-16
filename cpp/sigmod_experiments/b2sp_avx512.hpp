#include <immintrin.h>

#include <cstddef>
#include <cstdint>

inline int64_t popcount_512(__m512i bits) {
    const __m512i counts = _mm512_popcnt_epi64(bits);
    return _mm512_reduce_add_epi64(counts);
}

inline int64_t b2sp(
    __m512i* a_pos, 
    __m512i* a_neg, 
    __m512i* b_pos, 
    __m512i* b_neg, 
    size_t dim
) {
    int64_t result = 0;

    // Loops over each of the 512-bit widths.
    for (size_t i = 0; i < (dim + 511) / 512; ++i) {
        auto poses = _mm512_and_si512(a_pos[i], b_pos[i]);
        auto negs = _mm512_and_si512(a_neg[i], b_neg[i]);

        auto diff1 = _mm512_and_si512(a_pos[i], b_neg[i]);
        auto diff2 = _mm512_and_si512(b_pos[i], a_neg[i]);

        result += popcount_512(poses) + popcount_512(negs) - 
                  popcount_512(diff1) - popcount_512(diff2);
    }
    return result;
}

// The new set-sign AVX512 version
inline int64_t set_sign_b2sp(
    __m512i* a_set,
    __m512i* a_sign,
    __m512i* b_set,
    __m512i* b_sign,
    size_t dim
) {
    int64_t result = 0;

    // Loops over each of the 512-bit widths.
    for (size_t i = 0; i < (dim + 511) / 512; ++i) {
        auto overlap = _mm512_and_si512(a_set[i], b_set[i]);
        auto diff = _mm512_and_si512(overlap, _mm512_xor_si512(a_sign[i], b_sign[i]));
        result += popcount_512(overlap) - 2 * popcount_512(diff);
    }
    return result;
}
