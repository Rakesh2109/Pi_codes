#if defined(GLADE_V2_PROFILE) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "glade_v2.h"

#include <stddef.h>
#include <string.h>

#if defined(GLADE_V2_PROFILE)
#include <time.h>
#endif

#if defined(__AVX2__) || defined(__BMI2__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define GLADE_MIXED_FEATURE UINT32_MAX

enum {
    GLADE_PROFILE_TOTAL_CALLS = 0,
    GLADE_PROFILE_U8_CALLS,
    GLADE_PROFILE_PACKED_CALLS,
    GLADE_PROFILE_ROWS,
    GLADE_PROFILE_BITS,
    GLADE_PROFILE_OUTPUT_BYTES,
    GLADE_PROFILE_VALIDATE_NS,
    GLADE_PROFILE_KERNEL_NS,
    GLADE_PROFILE_TOTAL_NS,
    GLADE_PROFILE_ERRORS,
    GLADE_PROFILE_FIELD_COUNT
};

static const char *const GLADE_PROFILE_NAMES[GLADE_PROFILE_FIELD_COUNT] = {
    "total_calls",
    "u8_calls",
    "packed_calls",
    "rows",
    "bits",
    "output_bytes",
    "validate_ns",
    "kernel_ns",
    "total_ns",
    "errors",
};

#if defined(GLADE_V2_PROFILE)
static uint64_t glade_profile_values[GLADE_PROFILE_FIELD_COUNT];

static uint64_t profile_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void profile_add(uint32_t index, uint64_t value) {
    glade_profile_values[index] += value;
}

static void profile_error(void) {
    glade_profile_values[GLADE_PROFILE_ERRORS]++;
}
#else
static void profile_error(void) {
}
#endif

static int validate_features(
    const uint32_t *feature_indices,
    uint32_t n_features,
    uint32_t n_bits
) {
    for (uint32_t bit = 0; bit < n_bits; bit++) {
        if (feature_indices[bit] >= n_features) {
            return -1;
        }
    }
    return 0;
}

static inline uint8_t cmp_scalar(
    const double *row,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t bit
) {
    return (uint8_t)(row[feature_indices[bit]] >= thresholds[bit]);
}

static inline uint8_t cmp_scalar_f32(
    const float *row,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t bit
) {
    return (uint8_t)(row[feature_indices[bit]] >= thresholds[bit]);
}

static const uint8_t GLADE_PACK8_TABLE[256] = {
    0x00u, 0x80u, 0x40u, 0xc0u, 0x20u, 0xa0u, 0x60u, 0xe0u,
    0x10u, 0x90u, 0x50u, 0xd0u, 0x30u, 0xb0u, 0x70u, 0xf0u,
    0x08u, 0x88u, 0x48u, 0xc8u, 0x28u, 0xa8u, 0x68u, 0xe8u,
    0x18u, 0x98u, 0x58u, 0xd8u, 0x38u, 0xb8u, 0x78u, 0xf8u,
    0x04u, 0x84u, 0x44u, 0xc4u, 0x24u, 0xa4u, 0x64u, 0xe4u,
    0x14u, 0x94u, 0x54u, 0xd4u, 0x34u, 0xb4u, 0x74u, 0xf4u,
    0x0cu, 0x8cu, 0x4cu, 0xccu, 0x2cu, 0xacu, 0x6cu, 0xecu,
    0x1cu, 0x9cu, 0x5cu, 0xdcu, 0x3cu, 0xbcu, 0x7cu, 0xfcu,
    0x02u, 0x82u, 0x42u, 0xc2u, 0x22u, 0xa2u, 0x62u, 0xe2u,
    0x12u, 0x92u, 0x52u, 0xd2u, 0x32u, 0xb2u, 0x72u, 0xf2u,
    0x0au, 0x8au, 0x4au, 0xcau, 0x2au, 0xaau, 0x6au, 0xeau,
    0x1au, 0x9au, 0x5au, 0xdau, 0x3au, 0xbau, 0x7au, 0xfau,
    0x06u, 0x86u, 0x46u, 0xc6u, 0x26u, 0xa6u, 0x66u, 0xe6u,
    0x16u, 0x96u, 0x56u, 0xd6u, 0x36u, 0xb6u, 0x76u, 0xf6u,
    0x0eu, 0x8eu, 0x4eu, 0xceu, 0x2eu, 0xaeu, 0x6eu, 0xeeu,
    0x1eu, 0x9eu, 0x5eu, 0xdeu, 0x3eu, 0xbeu, 0x7eu, 0xfeu,
    0x01u, 0x81u, 0x41u, 0xc1u, 0x21u, 0xa1u, 0x61u, 0xe1u,
    0x11u, 0x91u, 0x51u, 0xd1u, 0x31u, 0xb1u, 0x71u, 0xf1u,
    0x09u, 0x89u, 0x49u, 0xc9u, 0x29u, 0xa9u, 0x69u, 0xe9u,
    0x19u, 0x99u, 0x59u, 0xd9u, 0x39u, 0xb9u, 0x79u, 0xf9u,
    0x05u, 0x85u, 0x45u, 0xc5u, 0x25u, 0xa5u, 0x65u, 0xe5u,
    0x15u, 0x95u, 0x55u, 0xd5u, 0x35u, 0xb5u, 0x75u, 0xf5u,
    0x0du, 0x8du, 0x4du, 0xcdu, 0x2du, 0xadu, 0x6du, 0xedu,
    0x1du, 0x9du, 0x5du, 0xddu, 0x3du, 0xbdu, 0x7du, 0xfdu,
    0x03u, 0x83u, 0x43u, 0xc3u, 0x23u, 0xa3u, 0x63u, 0xe3u,
    0x13u, 0x93u, 0x53u, 0xd3u, 0x33u, 0xb3u, 0x73u, 0xf3u,
    0x0bu, 0x8bu, 0x4bu, 0xcbu, 0x2bu, 0xabu, 0x6bu, 0xebu,
    0x1bu, 0x9bu, 0x5bu, 0xdbu, 0x3bu, 0xbbu, 0x7bu, 0xfbu,
    0x07u, 0x87u, 0x47u, 0xc7u, 0x27u, 0xa7u, 0x67u, 0xe7u,
    0x17u, 0x97u, 0x57u, 0xd7u, 0x37u, 0xb7u, 0x77u, 0xf7u,
    0x0fu, 0x8fu, 0x4fu, 0xcfu, 0x2fu, 0xafu, 0x6fu, 0xefu,
    0x1fu, 0x9fu, 0x5fu, 0xdfu, 0x3fu, 0xbfu, 0x7fu, 0xffu,
};

static const uint32_t GLADE_EXPAND_NIBBLE[16] = {
    0x00000000u, 0x00000001u, 0x00000100u, 0x00000101u,
    0x00010000u, 0x00010001u, 0x00010100u, 0x00010101u,
    0x01000000u, 0x01000001u, 0x01000100u, 0x01000101u,
    0x01010000u, 0x01010001u, 0x01010100u, 0x01010101u,
};

static inline uint8_t pack8_from_mask(unsigned mask) {
    return GLADE_PACK8_TABLE[mask & 0xffu];
}

static inline uint64_t expand8_mask_to_u8x8(unsigned mask) {
#if defined(__BMI2__)
    return _pdep_u64((uint64_t)(mask & 0xffu), UINT64_C(0x0101010101010101));
#else
    return (uint64_t)GLADE_EXPAND_NIBBLE[mask & 0x0fu] |
           ((uint64_t)GLADE_EXPAND_NIBBLE[(mask >> 4u) & 0x0fu] << 32u);
#endif
}

static inline void store_u8x8_from_mask(uint8_t *out, unsigned mask) {
    const uint64_t bytes = expand8_mask_to_u8x8(mask);
    memcpy(out, &bytes, sizeof(bytes));
}

static inline void store_u8x4_from_mask(uint8_t *out, unsigned mask) {
    const uint32_t bytes = GLADE_EXPAND_NIBBLE[mask & 0x0fu];
    memcpy(out, &bytes, sizeof(bytes));
}

static int validate_chunk_features4(
    const uint32_t *chunk_features4,
    uint32_t n_features,
    uint32_t n_bits
) {
    const uint32_t n_chunks4 = n_bits >> 2u;
    for (uint32_t chunk = 0; chunk < n_chunks4; chunk++) {
        if (
            chunk_features4[chunk] != GLADE_MIXED_FEATURE &&
            chunk_features4[chunk] >= n_features
        ) {
            return -1;
        }
    }
    return 0;
}

static int validate_chunk_features8(
    const uint32_t *chunk_features8,
    uint32_t n_features,
    uint32_t n_bits
) {
    const uint32_t n_chunks8 = n_bits >> 3u;
    for (uint32_t chunk = 0; chunk < n_chunks8; chunk++) {
        if (
            chunk_features8[chunk] != GLADE_MIXED_FEATURE &&
            chunk_features8[chunk] >= n_features
        ) {
            return -1;
        }
    }
    return 0;
}

#if defined(__AVX2__)
static inline unsigned cmp4_mask_avx2(
    const double *row,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t bit
) {
    const uint32_t f0 = feature_indices[bit];
    if (
        feature_indices[bit + 1u] == f0 &&
        feature_indices[bit + 2u] == f0 &&
        feature_indices[bit + 3u] == f0
    ) {
        const __m256d x = _mm256_set1_pd(row[f0]);
        const __m256d t = _mm256_loadu_pd(thresholds + bit);
        const __m256d ge = _mm256_cmp_pd(x, t, _CMP_GE_OQ);
        return (unsigned)_mm256_movemask_pd(ge);
    }
    const __m128i idx = _mm_loadu_si128((const __m128i *)(feature_indices + bit));
    const __m256d x = _mm256_i32gather_pd(row, idx, 8);
    const __m256d t = _mm256_loadu_pd(thresholds + bit);
    const __m256d ge = _mm256_cmp_pd(x, t, _CMP_GE_OQ);
    return (unsigned)_mm256_movemask_pd(ge);
}

static inline unsigned cmp8_mask_avx2_f32(
    const float *row,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t bit
) {
    const uint32_t f0 = feature_indices[bit];
    if (
        feature_indices[bit + 1u] == f0 &&
        feature_indices[bit + 2u] == f0 &&
        feature_indices[bit + 3u] == f0 &&
        feature_indices[bit + 4u] == f0 &&
        feature_indices[bit + 5u] == f0 &&
        feature_indices[bit + 6u] == f0 &&
        feature_indices[bit + 7u] == f0
    ) {
        const __m256 x = _mm256_set1_ps(row[f0]);
        const __m256 t = _mm256_loadu_ps(thresholds + bit);
        const __m256 ge = _mm256_cmp_ps(x, t, _CMP_GE_OQ);
        return (unsigned)_mm256_movemask_ps(ge);
    }
    const __m256i idx = _mm256_loadu_si256((const __m256i *)(feature_indices + bit));
    const __m256 x = _mm256_i32gather_ps(row, idx, 4);
    const __m256 t = _mm256_loadu_ps(thresholds + bit);
    const __m256 ge = _mm256_cmp_ps(x, t, _CMP_GE_OQ);
    return (unsigned)_mm256_movemask_ps(ge);
}

static inline unsigned cmp8_mask_avx2_f32_chunked(
    const float *row,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t bit
) {
    const uint32_t f0 = chunk_features8[bit >> 3u];
    if (f0 != GLADE_MIXED_FEATURE) {
        const __m256 x = _mm256_set1_ps(row[f0]);
        const __m256 t = _mm256_loadu_ps(thresholds + bit);
        const __m256 ge = _mm256_cmp_ps(x, t, _CMP_GE_OQ);
        return (unsigned)_mm256_movemask_ps(ge);
    }
    const __m256i idx = _mm256_loadu_si256((const __m256i *)(feature_indices + bit));
    const __m256 x = _mm256_i32gather_ps(row, idx, 4);
    const __m256 t = _mm256_loadu_ps(thresholds + bit);
    const __m256 ge = _mm256_cmp_ps(x, t, _CMP_GE_OQ);
    return (unsigned)_mm256_movemask_ps(ge);
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline unsigned mask4_u32_neon(uint32x4_t ge) {
    static const uint32_t mask_bits_data[4] = {1u, 2u, 4u, 8u};
    const uint32x4_t mask_bits = vld1q_u32(mask_bits_data);
    return (unsigned)vaddvq_u32(vandq_u32(ge, mask_bits));
}

static inline unsigned cmp2_mask_neon(
    const double *row,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t bit
) {
    const uint32_t f0 = feature_indices[bit];
    if (feature_indices[bit + 1u] == f0) {
        const float64x2_t x = vdupq_n_f64(row[f0]);
        const float64x2_t t = vld1q_f64(thresholds + bit);
        const uint64x2_t ge = vcgeq_f64(x, t);
        return ((unsigned)(vgetq_lane_u64(ge, 0) != 0u)) |
               ((unsigned)(vgetq_lane_u64(ge, 1) != 0u) << 1u);
    }
    const double values[2] = {
        row[feature_indices[bit]],
        row[feature_indices[bit + 1u]],
    };
    const float64x2_t x = vld1q_f64(values);
    const float64x2_t t = vld1q_f64(thresholds + bit);
    const uint64x2_t ge = vcgeq_f64(x, t);
    return ((unsigned)(vgetq_lane_u64(ge, 0) != 0u)) |
           ((unsigned)(vgetq_lane_u64(ge, 1) != 0u) << 1u);
}

static inline unsigned cmp4_mask_neon_f32(
    const float *row,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t bit
) {
    const uint32_t f0 = feature_indices[bit];
    if (
        feature_indices[bit + 1u] == f0 &&
        feature_indices[bit + 2u] == f0 &&
        feature_indices[bit + 3u] == f0
    ) {
        const float32x4_t x = vdupq_n_f32(row[f0]);
        const float32x4_t t = vld1q_f32(thresholds + bit);
        const uint32x4_t ge = vcgeq_f32(x, t);
        return mask4_u32_neon(ge);
    }
    return ((unsigned)(row[feature_indices[bit]] >= thresholds[bit])) |
           ((unsigned)(row[feature_indices[bit + 1u]] >= thresholds[bit + 1u]) << 1u) |
           ((unsigned)(row[feature_indices[bit + 2u]] >= thresholds[bit + 2u]) << 2u) |
           ((unsigned)(row[feature_indices[bit + 3u]] >= thresholds[bit + 3u]) << 3u);
}

static inline unsigned cmp4_mask_neon_f32_chunked(
    const float *row,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const float *thresholds,
    uint32_t bit
) {
    const uint32_t f0 = chunk_features4[bit >> 2u];
    if (f0 != GLADE_MIXED_FEATURE) {
        const float32x4_t x = vdupq_n_f32(row[f0]);
        const float32x4_t t = vld1q_f32(thresholds + bit);
        const uint32x4_t ge = vcgeq_f32(x, t);
        return mask4_u32_neon(ge);
    }
    return ((unsigned)(row[feature_indices[bit]] >= thresholds[bit])) |
           ((unsigned)(row[feature_indices[bit + 1u]] >= thresholds[bit + 1u]) << 1u) |
           ((unsigned)(row[feature_indices[bit + 2u]] >= thresholds[bit + 2u]) << 2u) |
           ((unsigned)(row[feature_indices[bit + 3u]] >= thresholds[bit + 3u]) << 3u);
}

#endif

static void transform_row_u8(
    const double *row,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
    uint32_t bit = 0;

#if defined(__AVX2__)
    for (; bit + 4u <= n_bits; bit += 4u) {
        const unsigned mask = cmp4_mask_avx2(row, feature_indices, thresholds, bit);
        out[bit] = (uint8_t)(mask & 1u);
        out[bit + 1u] = (uint8_t)((mask >> 1u) & 1u);
        out[bit + 2u] = (uint8_t)((mask >> 2u) & 1u);
        out[bit + 3u] = (uint8_t)((mask >> 3u) & 1u);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (; bit + 2u <= n_bits; bit += 2u) {
        const unsigned mask = cmp2_mask_neon(row, feature_indices, thresholds, bit);
        out[bit] = (uint8_t)(mask & 1u);
        out[bit + 1u] = (uint8_t)((mask >> 1u) & 1u);
    }
#endif

    for (; bit < n_bits; bit++) {
        out[bit] = cmp_scalar(row, feature_indices, thresholds, bit);
    }
}

static void transform_row_u8_f32(
    const float *row,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
    uint32_t bit = 0;

#if defined(__AVX2__)
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned mask = cmp8_mask_avx2_f32(row, feature_indices, thresholds, bit);
        store_u8x8_from_mask(out + bit, mask);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (; bit + 4u <= n_bits; bit += 4u) {
        const unsigned mask = cmp4_mask_neon_f32(row, feature_indices, thresholds, bit);
        store_u8x4_from_mask(out + bit, mask);
    }
#endif

    for (; bit < n_bits; bit++) {
        out[bit] = cmp_scalar_f32(row, feature_indices, thresholds, bit);
    }
}

static void transform_row_u8_chunked_f32(
    const float *row,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
    uint32_t bit = 0;

#if defined(__AVX2__)
    (void)chunk_features4;
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned mask =
            cmp8_mask_avx2_f32_chunked(row, feature_indices, chunk_features8, thresholds, bit);
        store_u8x8_from_mask(out + bit, mask);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    (void)chunk_features8;
    for (; bit + 4u <= n_bits; bit += 4u) {
        const unsigned mask =
            cmp4_mask_neon_f32_chunked(row, feature_indices, chunk_features4, thresholds, bit);
        store_u8x4_from_mask(out + bit, mask);
    }
#endif

    for (; bit < n_bits; bit++) {
        out[bit] = cmp_scalar_f32(row, feature_indices, thresholds, bit);
    }
}

static void transform_row_u8_chunks4_f32(
    const float *row,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_indices4,
    const float *thresholds,
    const float *chunk_thresholds4,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    const uint32_t n_chunks4 = n_bits >> 2u;
    for (uint32_t chunk = 0; chunk < n_chunks4; chunk++) {
        const uint32_t f0 = chunk_features4[chunk];
        const float *restrict t = chunk_thresholds4 + (size_t)chunk * 4u;
        unsigned mask;
        if (f0 != GLADE_MIXED_FEATURE) {
            const float32x4_t x = vdupq_n_f32(row[f0]);
            mask = mask4_u32_neon(vcgeq_f32(x, vld1q_f32(t)));
        } else {
            const uint32_t *restrict idx = chunk_indices4 + (size_t)chunk * 4u;
            mask = ((unsigned)(row[idx[0]] >= t[0])) |
                   ((unsigned)(row[idx[1]] >= t[1]) << 1u) |
                   ((unsigned)(row[idx[2]] >= t[2]) << 2u) |
                   ((unsigned)(row[idx[3]] >= t[3]) << 3u);
        }
        store_u8x4_from_mask(out + (size_t)chunk * 4u, mask);
    }
    for (uint32_t bit = n_chunks4 << 2u; bit < n_bits; bit++) {
        out[bit] = cmp_scalar_f32(row, feature_indices, thresholds, bit);
    }
#else
    transform_row_u8_chunked_f32(
        row,
        feature_indices,
        chunk_features4,
        NULL,
        thresholds,
        n_bits,
        out
    );
    (void)chunk_indices4;
    (void)chunk_thresholds4;
#endif
}

static void transform_row_packed(
    const double *row,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
    uint32_t bit = 0;

#if defined(__AVX2__)
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned lo = cmp4_mask_avx2(row, feature_indices, thresholds, bit);
        const unsigned hi = cmp4_mask_avx2(row, feature_indices, thresholds, bit + 4u);
        out[bit >> 3u] =
            (uint8_t)(((lo & 1u) << 7u) |
                      (((lo >> 1u) & 1u) << 6u) |
                      (((lo >> 2u) & 1u) << 5u) |
                      (((lo >> 3u) & 1u) << 4u) |
                      ((hi & 1u) << 3u) |
                      (((hi >> 1u) & 1u) << 2u) |
                      (((hi >> 2u) & 1u) << 1u) |
                      ((hi >> 3u) & 1u));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned m0 = cmp2_mask_neon(row, feature_indices, thresholds, bit);
        const unsigned m1 = cmp2_mask_neon(row, feature_indices, thresholds, bit + 2u);
        const unsigned m2 = cmp2_mask_neon(row, feature_indices, thresholds, bit + 4u);
        const unsigned m3 = cmp2_mask_neon(row, feature_indices, thresholds, bit + 6u);
        out[bit >> 3u] =
            (uint8_t)(((m0 & 1u) << 7u) |
                      (((m0 >> 1u) & 1u) << 6u) |
                      ((m1 & 1u) << 5u) |
                      (((m1 >> 1u) & 1u) << 4u) |
                      ((m2 & 1u) << 3u) |
                      (((m2 >> 1u) & 1u) << 2u) |
                      ((m3 & 1u) << 1u) |
                      ((m3 >> 1u) & 1u));
    }
#endif

    if (bit < n_bits) {
        out[bit >> 3u] = 0u;
    }
    for (; bit < n_bits; bit++) {
        if (cmp_scalar(row, feature_indices, thresholds, bit)) {
            out[bit >> 3u] |= (uint8_t)(0x80u >> (bit & 7u));
        }
    }
}

static void transform_row_packed_f32(
    const float *row,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
    uint32_t bit = 0;

#if defined(__AVX2__)
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned mask = cmp8_mask_avx2_f32(row, feature_indices, thresholds, bit);
        out[bit >> 3u] = pack8_from_mask(mask);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned lo = cmp4_mask_neon_f32(row, feature_indices, thresholds, bit);
        const unsigned hi = cmp4_mask_neon_f32(row, feature_indices, thresholds, bit + 4u);
        out[bit >> 3u] = pack8_from_mask(lo | (hi << 4u));
    }
#endif

    if (bit < n_bits) {
        out[bit >> 3u] = 0u;
    }
    for (; bit < n_bits; bit++) {
        if (cmp_scalar_f32(row, feature_indices, thresholds, bit)) {
            out[bit >> 3u] |= (uint8_t)(0x80u >> (bit & 7u));
        }
    }
}

static void transform_row_packed_chunked_f32(
    const float *row,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
    uint32_t bit = 0;

#if defined(__AVX2__)
    (void)chunk_features4;
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned mask =
            cmp8_mask_avx2_f32_chunked(row, feature_indices, chunk_features8, thresholds, bit);
        out[bit >> 3u] = pack8_from_mask(mask);
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
    (void)chunk_features8;
    for (; bit + 8u <= n_bits; bit += 8u) {
        const unsigned lo =
            cmp4_mask_neon_f32_chunked(row, feature_indices, chunk_features4, thresholds, bit);
        const unsigned hi = cmp4_mask_neon_f32_chunked(
            row,
            feature_indices,
            chunk_features4,
            thresholds,
            bit + 4u
        );
        out[bit >> 3u] = pack8_from_mask(lo | (hi << 4u));
    }
#endif

    if (bit < n_bits) {
        out[bit >> 3u] = 0u;
    }
    for (; bit < n_bits; bit++) {
        if (cmp_scalar_f32(row, feature_indices, thresholds, bit)) {
            out[bit >> 3u] |= (uint8_t)(0x80u >> (bit & 7u));
        }
    }
}

static void transform_row_packed_chunks4_f32(
    const float *row,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_indices4,
    const float *thresholds,
    const float *chunk_thresholds4,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    uint32_t chunk = 0;
    const uint32_t n_chunks8 = n_bits >> 3u;
    for (; chunk < n_chunks8 * 2u; chunk += 2u) {
        unsigned masks[2];
        for (uint32_t j = 0; j < 2u; j++) {
            const uint32_t c = chunk + j;
            const uint32_t f0 = chunk_features4[c];
            const float *restrict t = chunk_thresholds4 + (size_t)c * 4u;
            if (f0 != GLADE_MIXED_FEATURE) {
                const float32x4_t x = vdupq_n_f32(row[f0]);
                masks[j] = mask4_u32_neon(vcgeq_f32(x, vld1q_f32(t)));
            } else {
                const uint32_t *restrict idx = chunk_indices4 + (size_t)c * 4u;
                masks[j] = ((unsigned)(row[idx[0]] >= t[0])) |
                           ((unsigned)(row[idx[1]] >= t[1]) << 1u) |
                           ((unsigned)(row[idx[2]] >= t[2]) << 2u) |
                           ((unsigned)(row[idx[3]] >= t[3]) << 3u);
            }
        }
        out[chunk >> 1u] = pack8_from_mask(masks[0] | (masks[1] << 4u));
    }
    uint32_t bit = chunk << 2u;
    if (bit < n_bits) {
        out[bit >> 3u] = 0u;
    }
    for (; bit < n_bits; bit++) {
        if (cmp_scalar_f32(row, feature_indices, thresholds, bit)) {
            out[bit >> 3u] |= (uint8_t)(0x80u >> (bit & 7u));
        }
    }
#else
    transform_row_packed_chunked_f32(
        row,
        feature_indices,
        chunk_features4,
        NULL,
        thresholds,
        n_bits,
        out
    );
    (void)chunk_indices4;
    (void)chunk_thresholds4;
#endif
}

const char *glade_v2_backend(void) {
#if defined(__AVX2__)
    return "avx2";
#elif defined(__aarch64__) && defined(__ARM_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

uint32_t glade_v2_profile_field_count(void) {
    return GLADE_PROFILE_FIELD_COUNT;
}

const char *glade_v2_profile_field_name(uint32_t index) {
    if (index >= GLADE_PROFILE_FIELD_COUNT) {
        return "";
    }
    return GLADE_PROFILE_NAMES[index];
}

int glade_v2_profile_enabled(void) {
#if defined(GLADE_V2_PROFILE)
    return 1;
#else
    return 0;
#endif
}

void glade_v2_profile_reset(void) {
#if defined(GLADE_V2_PROFILE)
    memset(glade_profile_values, 0, sizeof(glade_profile_values));
#endif
}

int glade_v2_profile_read(uint64_t *out, uint32_t capacity) {
    if (out == NULL) {
        return -1;
    }
    if (capacity < GLADE_PROFILE_FIELD_COUNT) {
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    memcpy(out, glade_profile_values, sizeof(glade_profile_values));
#else
    memset(out, 0, sizeof(uint64_t) * GLADE_PROFILE_FIELD_COUNT);
#endif
    return (int)GLADE_PROFILE_FIELD_COUNT;
}

int glade_v2_transform_u8(
    const double *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (rows == NULL || feature_indices == NULL || thresholds == NULL || out == NULL) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t validate_start = profile_now_ns();
#endif
    if (validate_features(feature_indices, n_features, n_bits) != 0) {
        profile_error();
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_VALIDATE_NS, profile_now_ns() - validate_start);
    const uint64_t kernel_start = profile_now_ns();
#endif

    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_u8(
            rows + (size_t)r * n_features,
            feature_indices,
            thresholds,
            n_bits,
            out + (size_t)r * n_bits
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_U8_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bits);
#endif
    return 0;
}

int glade_v2_transform_u8_f32(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (rows == NULL || feature_indices == NULL || thresholds == NULL || out == NULL) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t validate_start = profile_now_ns();
#endif
    if (validate_features(feature_indices, n_features, n_bits) != 0) {
        profile_error();
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_VALIDATE_NS, profile_now_ns() - validate_start);
    const uint64_t kernel_start = profile_now_ns();
#endif

    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_u8_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            thresholds,
            n_bits,
            out + (size_t)r * n_bits
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_U8_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bits);
#endif
    return 0;
}

int glade_v2_transform_u8_chunked_f32(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (
        rows == NULL || feature_indices == NULL || chunk_features4 == NULL ||
        chunk_features8 == NULL || thresholds == NULL || out == NULL
    ) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t validate_start = profile_now_ns();
#endif
    if (
        validate_features(feature_indices, n_features, n_bits) != 0 ||
        validate_chunk_features4(chunk_features4, n_features, n_bits) != 0 ||
        validate_chunk_features8(chunk_features8, n_features, n_bits) != 0
    ) {
        profile_error();
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_VALIDATE_NS, profile_now_ns() - validate_start);
    const uint64_t kernel_start = profile_now_ns();
#endif

    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_u8_chunked_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            chunk_features4,
            chunk_features8,
            thresholds,
            n_bits,
            out + (size_t)r * n_bits
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_U8_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bits);
#endif
    return 0;
}

int glade_v2_transform_u8_chunked_f32_unchecked(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (
        rows == NULL || feature_indices == NULL || chunk_features4 == NULL ||
        chunk_features8 == NULL || thresholds == NULL || out == NULL
    ) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif

    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_u8_chunked_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            chunk_features4,
            chunk_features8,
            thresholds,
            n_bits,
            out + (size_t)r * n_bits
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_U8_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bits);
#endif
    return 0;
}

int glade_v2_transform_u8_chunks4_f32_unchecked(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_indices4,
    const float *thresholds,
    const float *chunk_thresholds4,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    (void)n_features;
    if (
        rows == NULL || feature_indices == NULL || chunk_features4 == NULL ||
        chunk_indices4 == NULL || thresholds == NULL || chunk_thresholds4 == NULL || out == NULL
    ) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif
    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_u8_chunks4_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            chunk_features4,
            chunk_indices4,
            thresholds,
            chunk_thresholds4,
            n_bits,
            out + (size_t)r * n_bits
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_U8_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bits);
#endif
    return 0;
}

int glade_v2_transform_packed(
    const double *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (rows == NULL || feature_indices == NULL || thresholds == NULL || out == NULL) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t validate_start = profile_now_ns();
#endif
    if (validate_features(feature_indices, n_features, n_bits) != 0) {
        profile_error();
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_VALIDATE_NS, profile_now_ns() - validate_start);
#endif

    const uint32_t n_bytes = (n_bits + 7u) >> 3u;
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif
    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_packed(
            rows + (size_t)r * n_features,
            feature_indices,
            thresholds,
            n_bits,
            out + (size_t)r * n_bytes
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_PACKED_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bytes);
#endif
    return 0;
}

int glade_v2_transform_packed_chunked_f32_unchecked(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (
        rows == NULL || feature_indices == NULL || chunk_features4 == NULL ||
        chunk_features8 == NULL || thresholds == NULL || out == NULL
    ) {
        profile_error();
        return -1;
    }

    const uint32_t n_bytes = (n_bits + 7u) >> 3u;
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif
    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_packed_chunked_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            chunk_features4,
            chunk_features8,
            thresholds,
            n_bits,
            out + (size_t)r * n_bytes
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_PACKED_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bytes);
#endif
    return 0;
}

int glade_v2_transform_packed_chunks4_f32_unchecked(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_indices4,
    const float *thresholds,
    const float *chunk_thresholds4,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (
        rows == NULL || feature_indices == NULL || chunk_features4 == NULL ||
        chunk_indices4 == NULL || thresholds == NULL || chunk_thresholds4 == NULL || out == NULL
    ) {
        profile_error();
        return -1;
    }

    const uint32_t n_bytes = (n_bits + 7u) >> 3u;
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif
    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_packed_chunks4_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            chunk_features4,
            chunk_indices4,
            thresholds,
            chunk_thresholds4,
            n_bits,
            out + (size_t)r * n_bytes
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_PACKED_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bytes);
#endif
    return 0;
}

int glade_v2_transform_packed_f32(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (rows == NULL || feature_indices == NULL || thresholds == NULL || out == NULL) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t validate_start = profile_now_ns();
#endif
    if (validate_features(feature_indices, n_features, n_bits) != 0) {
        profile_error();
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_VALIDATE_NS, profile_now_ns() - validate_start);
#endif

    const uint32_t n_bytes = (n_bits + 7u) >> 3u;
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif
    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_packed_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            thresholds,
            n_bits,
            out + (size_t)r * n_bytes
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_PACKED_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bytes);
#endif
    return 0;
}

int glade_v2_transform_packed_chunked_f32(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const uint32_t *chunk_features4,
    const uint32_t *chunk_features8,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
) {
#if defined(GLADE_V2_PROFILE)
    const uint64_t total_start = profile_now_ns();
#endif
    if (
        rows == NULL || feature_indices == NULL || chunk_features4 == NULL ||
        chunk_features8 == NULL || thresholds == NULL || out == NULL
    ) {
        profile_error();
        return -1;
    }
#if defined(GLADE_V2_PROFILE)
    const uint64_t validate_start = profile_now_ns();
#endif
    if (
        validate_features(feature_indices, n_features, n_bits) != 0 ||
        validate_chunk_features4(chunk_features4, n_features, n_bits) != 0 ||
        validate_chunk_features8(chunk_features8, n_features, n_bits) != 0
    ) {
        profile_error();
        return -2;
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_VALIDATE_NS, profile_now_ns() - validate_start);
#endif

    const uint32_t n_bytes = (n_bits + 7u) >> 3u;
#if defined(GLADE_V2_PROFILE)
    const uint64_t kernel_start = profile_now_ns();
#endif
    for (uint32_t r = 0; r < n_rows; r++) {
        transform_row_packed_chunked_f32(
            rows + (size_t)r * n_features,
            feature_indices,
            chunk_features4,
            chunk_features8,
            thresholds,
            n_bits,
            out + (size_t)r * n_bytes
        );
    }
#if defined(GLADE_V2_PROFILE)
    profile_add(GLADE_PROFILE_KERNEL_NS, profile_now_ns() - kernel_start);
    profile_add(GLADE_PROFILE_TOTAL_NS, profile_now_ns() - total_start);
    profile_add(GLADE_PROFILE_TOTAL_CALLS, 1u);
    profile_add(GLADE_PROFILE_PACKED_CALLS, 1u);
    profile_add(GLADE_PROFILE_ROWS, n_rows);
    profile_add(GLADE_PROFILE_BITS, (uint64_t)n_rows * (uint64_t)n_bits);
    profile_add(GLADE_PROFILE_OUTPUT_BYTES, (uint64_t)n_rows * (uint64_t)n_bytes);
#endif
    return 0;
}
