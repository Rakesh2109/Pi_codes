#include "tm_kernel_17.h"

/*
 * Attempt 17 SIMD scoring kernels.
 *
 * The kernels evaluate selected byte/delta table rows for fixed class-block
 * shapes. Keep this file focused on scoring; binarization and CLI/cache logic
 * belong in tm_algorithm.c and tm_infer_c.c.
 */

#include <limits.h>
#include <stddef.h>
#include <string.h>

#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
#include <arm_neon.h>
#endif

#define TM17_AVX2_LANES 32u
#define TM17_NEON_LANES 16u

typedef struct {
    uint32_t pos_start;
    uint32_t pos_end;
    uint32_t neg_start;
    uint32_t neg_end;
} TM17ClassRange;

static inline uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static inline uint32_t block_count(uint32_t start, uint32_t end, uint32_t lanes) {
    return min_u32(end - start, lanes);
}

static inline uint32_t positive_count(uint32_t start, uint32_t count, uint32_t pos_end) {
    return start < pos_end ? min_u32(pos_end - start, count) : 0u;
}

static inline TM17ClassRange class_range(const TMLayout *restrict layout, uint16_t k) {
    TM17ClassRange r;
    r.pos_start = (uint32_t)layout->pos_start[k];
    r.pos_end = (uint32_t)layout->pos_end[k];
    r.neg_start = (uint32_t)layout->neg_start[k];
    r.neg_end = (uint32_t)layout->neg_end[k];
    return r;
}

static inline int use_delta_table(const TMLayout *restrict layout) {
    return layout->prefer_delta_table != 0u &&
           layout->delta_table_exact != 0u &&
           layout->byte_delta_tables != NULL &&
           layout->base_mismatch_zero != NULL;
}

static inline void record_vote(
    uint16_t k,
    int32_t vote,
    int32_t *restrict votes,
    int32_t *restrict best_vote,
    int32_t *restrict best_class
) {
    votes[k] = vote;
    if (vote > *best_vote) {
        *best_vote = vote;
        *best_class = (int32_t)k;
    }
}

static inline void current_to_bytes(
    const uint64_t *restrict current,
    uint8_t *restrict current_bytes,
    uint32_t n_bytes
) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    memcpy(current_bytes, current, n_bytes);
#else
    for (uint32_t b = 0; b < n_bytes; b++) {
        current_bytes[b] = (uint8_t)(current[b >> 3] >> (8u * (b & 7u)));
    }
#endif
}

static inline void select_table_rows(
    const uint8_t *restrict tables,
    uint32_t stride,
    const uint8_t *restrict current_bytes,
    uint32_t n_bytes,
    const uint8_t **restrict rows
) {
    for (uint32_t b = 0; b < n_bytes; b++) {
        rows[b] = tables + ((size_t)b * 256u + current_bytes[b]) * stride;
    }
}

static inline uint32_t select_nonzero_delta_rows(
    const int8_t *restrict tables,
    uint32_t stride,
    const uint8_t *restrict current_bytes,
    uint32_t n_bytes,
    const int8_t **restrict rows
) {
    uint32_t n = 0;
    for (uint32_t b = 0; b < n_bytes; b++) {
        const uint8_t value = current_bytes[b];
        if (value != 0u) {
            rows[n++] = tables + ((size_t)b * 256u + value) * stride;
        }
    }
    return n;
}

#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
static inline int32_t hsum_u8x32(__m256i v, __m256i zero) {
    const __m256i sad = _mm256_sad_epu8(v, zero);
    const __m128i lo = _mm256_castsi256_si128(sad);
    const __m128i hi = _mm256_extracti128_si256(sad, 1);
    const __m128i sum = _mm_add_epi64(lo, hi);
    return (int32_t)((uint64_t)_mm_cvtsi128_si64(sum) + (uint64_t)_mm_extract_epi64(sum, 1));
}

static inline __m256i tail_mask_avx2(uint32_t count, __m256i lane_ids) {
    return _mm256_cmpgt_epi8(_mm256_set1_epi8((char)count), lane_ids);
}

static inline void accum_vote_avx2(
    uint32_t c,
    __m256i mismatches,
    const TM17ClassRange *restrict r,
    const uint8_t *restrict clamp,
    __m256i zero,
    __m256i lane_ids,
    int32_t *restrict vote
) {
    const uint32_t count = block_count(c, r->neg_end, TM17_AVX2_LANES);
    __m256i out = _mm256_subs_epu8(
        _mm256_loadu_si256((const __m256i *)(clamp + c)),
        mismatches
    );
    if (count < TM17_AVX2_LANES) {
        out = _mm256_and_si256(out, tail_mask_avx2(count, lane_ids));
    }

    const uint32_t pos_count = positive_count(c, count, r->pos_end);
    const int32_t total = hsum_u8x32(out, zero);
#if !defined(TM17_SIMPLE_REDUCE)
    if (pos_count == 0u) {
        *vote -= total;
    } else if (pos_count == count) {
        *vote += total;
    } else {
        const int32_t pos = hsum_u8x32(_mm256_and_si256(out, tail_mask_avx2(pos_count, lane_ids)), zero);
        *vote += 2 * pos - total;
    }
#else
    const int32_t pos = pos_count
        ? hsum_u8x32(_mm256_and_si256(out, tail_mask_avx2(pos_count, lane_ids)), zero)
        : 0;
    *vote += 2 * pos - total;
#endif
}

#define TM17_AVX2_TABLE_N(NAME, N)                                                              \
    static int32_t NAME(                                                                         \
        const TMLayout *restrict layout,                                                         \
        const uint8_t *const *restrict rows,                                                     \
        uint16_t k,                                                                              \
        __m256i zero,                                                                            \
        __m256i lane_ids                                                                         \
    ) {                                                                                          \
        const TM17ClassRange r = class_range(layout, k);                                         \
        const uint32_t c0 = r.pos_start;                                                         \
        const uint32_t c1 = c0 + TM17_AVX2_LANES;                                                \
        const uint32_t c2 = c1 + TM17_AVX2_LANES;                                                \
        const uint32_t c3 = c2 + TM17_AVX2_LANES;                                                \
        __m256i m0 = zero;                                                                       \
        __m256i m1 = zero;                                                                       \
        __m256i m2 = zero;                                                                       \
        __m256i m3 = zero;                                                                       \
        int32_t vote = 0;                                                                        \
        for (uint32_t b = 0; b < layout->n_bytes; b++) {                                         \
            const uint8_t *restrict row = rows[b];                                               \
            m0 = _mm256_adds_epu8(m0, _mm256_loadu_si256((const __m256i *)(row + c0)));          \
            m1 = _mm256_adds_epu8(m1, _mm256_loadu_si256((const __m256i *)(row + c1)));          \
            if ((N) >= 3) {                                                                      \
                m2 = _mm256_adds_epu8(m2, _mm256_loadu_si256((const __m256i *)(row + c2)));      \
            }                                                                                    \
            if ((N) >= 4) {                                                                      \
                m3 = _mm256_adds_epu8(m3, _mm256_loadu_si256((const __m256i *)(row + c3)));      \
            }                                                                                    \
        }                                                                                        \
        accum_vote_avx2(c0, m0, &r, layout->clamp, zero, lane_ids, &vote);                       \
        accum_vote_avx2(c1, m1, &r, layout->clamp, zero, lane_ids, &vote);                       \
        if ((N) >= 3) {                                                                          \
            accum_vote_avx2(c2, m2, &r, layout->clamp, zero, lane_ids, &vote);                   \
        }                                                                                        \
        if ((N) >= 4) {                                                                          \
            accum_vote_avx2(c3, m3, &r, layout->clamp, zero, lane_ids, &vote);                   \
        }                                                                                        \
        return vote;                                                                             \
    }

TM17_AVX2_TABLE_N(score_class_table2_avx2, 2)
TM17_AVX2_TABLE_N(score_class_table3_avx2, 3)
TM17_AVX2_TABLE_N(score_class_table4_avx2, 4)

#define TM17_AVX2_DELTA_N(NAME, N)                                                              \
    static int32_t NAME(                                                                         \
        const TMLayout *restrict layout,                                                         \
        const int8_t *const *restrict rows,                                                      \
        uint32_t n_rows,                                                                         \
        uint16_t k,                                                                              \
        __m256i zero,                                                                            \
        __m256i lane_ids                                                                         \
    ) {                                                                                          \
        const TM17ClassRange r = class_range(layout, k);                                         \
        const uint32_t c0 = r.pos_start;                                                         \
        const uint32_t c1 = c0 + TM17_AVX2_LANES;                                                \
        const uint32_t c2 = c1 + TM17_AVX2_LANES;                                                \
        const uint32_t c3 = c2 + TM17_AVX2_LANES;                                                \
        __m256i m0 = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c0));     \
        __m256i m1 = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c1));     \
        __m256i m2 = (N) >= 3                                                                    \
            ? _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c2))             \
            : zero;                                                                              \
        __m256i m3 = (N) >= 4                                                                    \
            ? _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c3))             \
            : zero;                                                                              \
        int32_t vote = 0;                                                                        \
        for (uint32_t i = 0; i < n_rows; i++) {                                                  \
            const int8_t *restrict row = rows[i];                                                \
            m0 = _mm256_add_epi8(m0, _mm256_loadu_si256((const __m256i *)(row + c0)));           \
            m1 = _mm256_add_epi8(m1, _mm256_loadu_si256((const __m256i *)(row + c1)));           \
            if ((N) >= 3) {                                                                      \
                m2 = _mm256_add_epi8(m2, _mm256_loadu_si256((const __m256i *)(row + c2)));       \
            }                                                                                    \
            if ((N) >= 4) {                                                                      \
                m3 = _mm256_add_epi8(m3, _mm256_loadu_si256((const __m256i *)(row + c3)));       \
            }                                                                                    \
        }                                                                                        \
        accum_vote_avx2(c0, m0, &r, layout->clamp, zero, lane_ids, &vote);                       \
        accum_vote_avx2(c1, m1, &r, layout->clamp, zero, lane_ids, &vote);                       \
        if ((N) >= 3) {                                                                          \
            accum_vote_avx2(c2, m2, &r, layout->clamp, zero, lane_ids, &vote);                   \
        }                                                                                        \
        if ((N) >= 4) {                                                                          \
            accum_vote_avx2(c3, m3, &r, layout->clamp, zero, lane_ids, &vote);                   \
        }                                                                                        \
        return vote;                                                                             \
    }

TM17_AVX2_DELTA_N(score_class_delta2_avx2, 2)
TM17_AVX2_DELTA_N(score_class_delta3_avx2, 3)
TM17_AVX2_DELTA_N(score_class_delta4_avx2, 4)

static int32_t score_class_table_generic_avx2(
    const TMLayout *restrict layout,
    const uint8_t *const *restrict rows,
    uint16_t k,
    __m256i zero,
    __m256i lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    int32_t vote = 0;
    uint32_t c = r.pos_start;
    for (; c + 4u * TM17_AVX2_LANES <= r.neg_end; c += 4u * TM17_AVX2_LANES) {
        const uint32_t c0 = c;
        const uint32_t c1 = c0 + TM17_AVX2_LANES;
        const uint32_t c2 = c1 + TM17_AVX2_LANES;
        const uint32_t c3 = c2 + TM17_AVX2_LANES;
        __m256i m0 = zero;
        __m256i m1 = zero;
        __m256i m2 = zero;
        __m256i m3 = zero;
        for (uint32_t b = 0; b < layout->n_bytes; b++) {
            const uint8_t *restrict row = rows[b];
            m0 = _mm256_adds_epu8(m0, _mm256_loadu_si256((const __m256i *)(row + c0)));
            m1 = _mm256_adds_epu8(m1, _mm256_loadu_si256((const __m256i *)(row + c1)));
            m2 = _mm256_adds_epu8(m2, _mm256_loadu_si256((const __m256i *)(row + c2)));
            m3 = _mm256_adds_epu8(m3, _mm256_loadu_si256((const __m256i *)(row + c3)));
        }
        accum_vote_avx2(c0, m0, &r, layout->clamp, zero, lane_ids, &vote);
        accum_vote_avx2(c1, m1, &r, layout->clamp, zero, lane_ids, &vote);
        accum_vote_avx2(c2, m2, &r, layout->clamp, zero, lane_ids, &vote);
        accum_vote_avx2(c3, m3, &r, layout->clamp, zero, lane_ids, &vote);
    }
    for (; c < r.neg_end; c += TM17_AVX2_LANES) {
        __m256i mismatches = zero;
        for (uint32_t b = 0; b < layout->n_bytes; b++) {
            mismatches = _mm256_adds_epu8(
                mismatches,
                _mm256_loadu_si256((const __m256i *)(rows[b] + c))
            );
        }
        accum_vote_avx2(c, mismatches, &r, layout->clamp, zero, lane_ids, &vote);
    }
    return vote;
}

static int32_t score_class_delta_generic_avx2(
    const TMLayout *restrict layout,
    const int8_t *const *restrict rows,
    uint32_t n_rows,
    uint16_t k,
    __m256i zero,
    __m256i lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    int32_t vote = 0;
    uint32_t c = r.pos_start;
    for (; c + 4u * TM17_AVX2_LANES <= r.neg_end; c += 4u * TM17_AVX2_LANES) {
        const uint32_t c0 = c;
        const uint32_t c1 = c0 + TM17_AVX2_LANES;
        const uint32_t c2 = c1 + TM17_AVX2_LANES;
        const uint32_t c3 = c2 + TM17_AVX2_LANES;
        __m256i m0 = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c0));
        __m256i m1 = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c1));
        __m256i m2 = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c2));
        __m256i m3 = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c3));
        for (uint32_t i = 0; i < n_rows; i++) {
            const int8_t *restrict row = rows[i];
            m0 = _mm256_add_epi8(m0, _mm256_loadu_si256((const __m256i *)(row + c0)));
            m1 = _mm256_add_epi8(m1, _mm256_loadu_si256((const __m256i *)(row + c1)));
            m2 = _mm256_add_epi8(m2, _mm256_loadu_si256((const __m256i *)(row + c2)));
            m3 = _mm256_add_epi8(m3, _mm256_loadu_si256((const __m256i *)(row + c3)));
        }
        accum_vote_avx2(c0, m0, &r, layout->clamp, zero, lane_ids, &vote);
        accum_vote_avx2(c1, m1, &r, layout->clamp, zero, lane_ids, &vote);
        accum_vote_avx2(c2, m2, &r, layout->clamp, zero, lane_ids, &vote);
        accum_vote_avx2(c3, m3, &r, layout->clamp, zero, lane_ids, &vote);
    }
    for (; c < r.neg_end; c += TM17_AVX2_LANES) {
        __m256i mismatches = _mm256_loadu_si256((const __m256i *)(layout->base_mismatch_zero + c));
        for (uint32_t i = 0; i < n_rows; i++) {
            mismatches = _mm256_add_epi8(
                mismatches,
                _mm256_loadu_si256((const __m256i *)(rows[i] + c))
            );
        }
        accum_vote_avx2(c, mismatches, &r, layout->clamp, zero, lane_ids, &vote);
    }
    return vote;
}

static int32_t score_class_table_avx2(
    const TMLayout *restrict layout,
    const uint8_t *const *restrict rows,
    uint16_t k,
    __m256i zero,
    __m256i lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    const uint32_t n_blocks = (r.neg_end - r.pos_start + TM17_AVX2_LANES - 1u) / TM17_AVX2_LANES;
    switch (n_blocks) {
        case 2u: return score_class_table2_avx2(layout, rows, k, zero, lane_ids);
        case 3u: return score_class_table3_avx2(layout, rows, k, zero, lane_ids);
        case 4u: return score_class_table4_avx2(layout, rows, k, zero, lane_ids);
        default: return score_class_table_generic_avx2(layout, rows, k, zero, lane_ids);
    }
}

static int32_t score_class_delta_avx2(
    const TMLayout *restrict layout,
    const int8_t *const *restrict rows,
    uint32_t n_rows,
    uint16_t k,
    __m256i zero,
    __m256i lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    const uint32_t n_blocks = (r.neg_end - r.pos_start + TM17_AVX2_LANES - 1u) / TM17_AVX2_LANES;
    switch (n_blocks) {
        case 2u: return score_class_delta2_avx2(layout, rows, n_rows, k, zero, lane_ids);
        case 3u: return score_class_delta3_avx2(layout, rows, n_rows, k, zero, lane_ids);
        case 4u: return score_class_delta4_avx2(layout, rows, n_rows, k, zero, lane_ids);
        default: return score_class_delta_generic_avx2(layout, rows, n_rows, k, zero, lane_ids);
    }
}

#if defined(TM_ENABLE_AOT_SHAPES)
#define TM17_AOT_AVX2_PRED(NAME, NBYTES, NCLASSES, TABLE_FN, DELTA_FN)                         \
    static int32_t NAME(                                                                        \
        const TMLayout *restrict layout,                                                        \
        const uint64_t *restrict current,                                                       \
        int32_t *restrict votes,                                                                \
        __m256i zero,                                                                           \
        __m256i lane_ids                                                                        \
    ) {                                                                                         \
        uint8_t current_bytes[TM_MAX_H_WORDS * 8u];                                             \
        const uint8_t *table_rows[TM_MAX_H_WORDS * 8u];                                         \
        const int8_t *delta_rows[TM_MAX_H_WORDS * 8u];                                          \
        const int use_delta = use_delta_table(layout);                                          \
        uint32_t n_delta = 0u;                                                                  \
        current_to_bytes(current, current_bytes, (NBYTES));                                     \
        if (use_delta) {                                                                        \
            n_delta = select_nonzero_delta_rows(                                                \
                layout->byte_delta_tables,                                                      \
                layout->table_stride,                                                           \
                current_bytes,                                                                  \
                (NBYTES),                                                                       \
                delta_rows                                                                      \
            );                                                                                  \
        } else {                                                                                \
            select_table_rows(layout->byte_tables, layout->table_stride, current_bytes, (NBYTES), table_rows); \
        }                                                                                       \
        int32_t best_vote = INT32_MIN / 2;                                                      \
        int32_t best_class = 0;                                                                 \
        if (use_delta) {                                                                        \
            for (uint16_t k = 0; k < (NCLASSES); k++) {                                         \
                const int32_t vote = DELTA_FN(layout, delta_rows, n_delta, k, zero, lane_ids);   \
                record_vote(k, vote, votes, &best_vote, &best_class);                           \
            }                                                                                   \
        } else {                                                                                \
            for (uint16_t k = 0; k < (NCLASSES); k++) {                                         \
                const int32_t vote = TABLE_FN(layout, table_rows, k, zero, lane_ids);            \
                record_vote(k, vote, votes, &best_vote, &best_class);                           \
            }                                                                                   \
        }                                                                                       \
        return best_class;                                                                      \
    }

TM17_AOT_AVX2_PRED(predict_wustl_avx2, 31u, 3u, score_class_table2_avx2, score_class_delta2_avx2)
TM17_AOT_AVX2_PRED(predict_nslkdd_avx2, 32u, 5u, score_class_table3_avx2, score_class_delta3_avx2)
TM17_AOT_AVX2_PRED(predict_toniot_avx2, 16u, 10u, score_class_table4_avx2, score_class_delta4_avx2)
TM17_AOT_AVX2_PRED(predict_medsec_avx2, 44u, 5u, score_class_table3_avx2, score_class_delta3_avx2)
#endif

int32_t tm17_predict_current_avx2(
    const TMLayout *restrict layout,
    const uint64_t *restrict current,
    int32_t *restrict votes
) {
    uint8_t current_bytes[TM_MAX_H_WORDS * 8u];
    const uint8_t *table_rows[TM_MAX_H_WORDS * 8u];
    const int8_t *delta_rows[TM_MAX_H_WORDS * 8u];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i lane_ids = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31
    );

#if defined(TM_ENABLE_AOT_SHAPES)
    if (layout->n_classes == 3u && layout->n_bytes == 31u && layout->n_clauses == 180u) {
        return predict_wustl_avx2(layout, current, votes, zero, lane_ids);
    }
    if (layout->n_classes == 5u && layout->n_bytes == 32u && layout->n_clauses == 450u) {
        return predict_nslkdd_avx2(layout, current, votes, zero, lane_ids);
    }
    if (layout->n_classes == 10u && layout->n_bytes == 16u && layout->n_clauses == 1000u) {
        return predict_toniot_avx2(layout, current, votes, zero, lane_ids);
    }
    if (layout->n_classes == 5u && layout->n_bytes == 44u && layout->n_clauses == 400u) {
        return predict_medsec_avx2(layout, current, votes, zero, lane_ids);
    }
#endif

    const int use_delta = use_delta_table(layout);
    uint32_t n_delta = 0u;

    if (use_delta) {
        current_to_bytes(current, current_bytes, layout->n_bytes);
        n_delta = select_nonzero_delta_rows(
            layout->byte_delta_tables,
            layout->table_stride,
            current_bytes,
            layout->n_bytes,
            delta_rows
        );
    } else {
        current_to_bytes(current, current_bytes, layout->n_bytes);
        select_table_rows(
            layout->byte_tables,
            layout->table_stride,
            current_bytes,
            layout->n_bytes,
            table_rows
        );
    }

    int32_t best_vote = INT32_MIN / 2;
    int32_t best_class = 0;
    for (uint16_t k = 0; k < layout->n_classes; k++) {
        const int32_t vote = use_delta
            ? score_class_delta_avx2(layout, delta_rows, n_delta, k, zero, lane_ids)
            : score_class_table_avx2(layout, table_rows, k, zero, lane_ids);
        record_vote(k, vote, votes, &best_vote, &best_class);
    }
    return best_class;
}
#endif

#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
static inline int32_t hsum_u8x16(uint8x16_t v) {
#if defined(__aarch64__)
    return (int32_t)vaddlvq_u8(v);
#else
    const uint16x8_t s16 = vpaddlq_u8(v);
    const uint32x4_t s32 = vpaddlq_u16(s16);
    const uint64x2_t s64 = vpaddlq_u32(s32);
    return (int32_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
}

static inline uint8x16_t tail_mask_neon(uint32_t count, uint8x16_t lane_ids) {
    return vcltq_u8(lane_ids, vdupq_n_u8((uint8_t)count));
}

static inline uint8x16_t add_i8_as_u8(uint8x16_t a, const int8_t *restrict b) {
    return vreinterpretq_u8_s8(vaddq_s8(vreinterpretq_s8_u8(a), vld1q_s8(b)));
}

static inline void accum_vote_neon(
    uint32_t c,
    uint8x16_t mismatches,
    const TM17ClassRange *restrict r,
    const uint8_t *restrict clamp,
    uint8x16_t lane_ids,
    int32_t *restrict vote
) {
    const uint32_t count = block_count(c, r->neg_end, TM17_NEON_LANES);
    uint8x16_t out = vqsubq_u8(vld1q_u8(clamp + c), mismatches);
    if (count < TM17_NEON_LANES) {
        out = vandq_u8(out, tail_mask_neon(count, lane_ids));
    }

    const uint32_t pos_count = positive_count(c, count, r->pos_end);
    const int32_t total = hsum_u8x16(out);
#if !defined(TM17_SIMPLE_REDUCE)
    if (pos_count == 0u) {
        *vote -= total;
    } else if (pos_count == count) {
        *vote += total;
    } else {
        const int32_t pos = hsum_u8x16(vandq_u8(out, tail_mask_neon(pos_count, lane_ids)));
        *vote += 2 * pos - total;
    }
#else
    const int32_t pos = pos_count
        ? hsum_u8x16(vandq_u8(out, tail_mask_neon(pos_count, lane_ids)))
        : 0;
    *vote += 2 * pos - total;
#endif
}

#define TM17_NEON_TABLE_N(NAME, N)                                                              \
    static int32_t NAME(                                                                         \
        const TMLayout *restrict layout,                                                         \
        const uint8_t *const *restrict rows,                                                     \
        uint16_t k,                                                                              \
        uint8x16_t zero,                                                                         \
        uint8x16_t lane_ids                                                                      \
    ) {                                                                                          \
        const TM17ClassRange r = class_range(layout, k);                                         \
        const uint32_t c0 = r.pos_start;                                                         \
        const uint32_t c1 = c0 + TM17_NEON_LANES;                                                \
        const uint32_t c2 = c1 + TM17_NEON_LANES;                                                \
        const uint32_t c3 = c2 + TM17_NEON_LANES;                                                \
        const uint32_t c4 = c3 + TM17_NEON_LANES;                                                \
        const uint32_t c5 = c4 + TM17_NEON_LANES;                                                \
        const uint32_t c6 = c5 + TM17_NEON_LANES;                                                \
        const uint32_t c7 = c6 + TM17_NEON_LANES;                                                \
        uint8x16_t m0 = zero;                                                                    \
        uint8x16_t m1 = zero;                                                                    \
        uint8x16_t m2 = zero;                                                                    \
        uint8x16_t m3 = zero;                                                                    \
        uint8x16_t m4 = zero;                                                                    \
        uint8x16_t m5 = zero;                                                                    \
        uint8x16_t m6 = zero;                                                                    \
        uint8x16_t m7 = zero;                                                                    \
        int32_t vote = 0;                                                                        \
        for (uint32_t b = 0; b < layout->n_bytes; b++) {                                         \
            const uint8_t *restrict row = rows[b];                                               \
            m0 = vqaddq_u8(m0, vld1q_u8(row + c0));                                             \
            m1 = vqaddq_u8(m1, vld1q_u8(row + c1));                                             \
            if ((N) >= 3) m2 = vqaddq_u8(m2, vld1q_u8(row + c2));                               \
            if ((N) >= 4) m3 = vqaddq_u8(m3, vld1q_u8(row + c3));                               \
            if ((N) >= 5) m4 = vqaddq_u8(m4, vld1q_u8(row + c4));                               \
            if ((N) >= 6) m5 = vqaddq_u8(m5, vld1q_u8(row + c5));                               \
            if ((N) >= 7) m6 = vqaddq_u8(m6, vld1q_u8(row + c6));                               \
            if ((N) >= 8) m7 = vqaddq_u8(m7, vld1q_u8(row + c7));                               \
        }                                                                                        \
        accum_vote_neon(c0, m0, &r, layout->clamp, lane_ids, &vote);                             \
        accum_vote_neon(c1, m1, &r, layout->clamp, lane_ids, &vote);                             \
        if ((N) >= 3) accum_vote_neon(c2, m2, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 4) accum_vote_neon(c3, m3, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 5) accum_vote_neon(c4, m4, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 6) accum_vote_neon(c5, m5, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 7) accum_vote_neon(c6, m6, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 8) accum_vote_neon(c7, m7, &r, layout->clamp, lane_ids, &vote);               \
        return vote;                                                                             \
    }

TM17_NEON_TABLE_N(score_class_table2_neon, 2)
TM17_NEON_TABLE_N(score_class_table3_neon, 3)
TM17_NEON_TABLE_N(score_class_table4_neon, 4)
TM17_NEON_TABLE_N(score_class_table5_neon, 5)
TM17_NEON_TABLE_N(score_class_table6_neon, 6)
TM17_NEON_TABLE_N(score_class_table7_neon, 7)
TM17_NEON_TABLE_N(score_class_table8_neon, 8)

#define TM17_NEON_DELTA_N(NAME, N)                                                              \
    static int32_t NAME(                                                                         \
        const TMLayout *restrict layout,                                                         \
        const int8_t *const *restrict rows,                                                      \
        uint32_t n_rows,                                                                         \
        uint16_t k,                                                                              \
        uint8x16_t zero,                                                                         \
        uint8x16_t lane_ids                                                                      \
    ) {                                                                                          \
        const TM17ClassRange r = class_range(layout, k);                                         \
        const uint32_t c0 = r.pos_start;                                                         \
        const uint32_t c1 = c0 + TM17_NEON_LANES;                                                \
        const uint32_t c2 = c1 + TM17_NEON_LANES;                                                \
        const uint32_t c3 = c2 + TM17_NEON_LANES;                                                \
        const uint32_t c4 = c3 + TM17_NEON_LANES;                                                \
        const uint32_t c5 = c4 + TM17_NEON_LANES;                                                \
        const uint32_t c6 = c5 + TM17_NEON_LANES;                                                \
        const uint32_t c7 = c6 + TM17_NEON_LANES;                                                \
        uint8x16_t m0 = vld1q_u8(layout->base_mismatch_zero + c0);                               \
        uint8x16_t m1 = vld1q_u8(layout->base_mismatch_zero + c1);                               \
        uint8x16_t m2 = (N) >= 3 ? vld1q_u8(layout->base_mismatch_zero + c2) : zero;             \
        uint8x16_t m3 = (N) >= 4 ? vld1q_u8(layout->base_mismatch_zero + c3) : zero;             \
        uint8x16_t m4 = (N) >= 5 ? vld1q_u8(layout->base_mismatch_zero + c4) : zero;             \
        uint8x16_t m5 = (N) >= 6 ? vld1q_u8(layout->base_mismatch_zero + c5) : zero;             \
        uint8x16_t m6 = (N) >= 7 ? vld1q_u8(layout->base_mismatch_zero + c6) : zero;             \
        uint8x16_t m7 = (N) >= 8 ? vld1q_u8(layout->base_mismatch_zero + c7) : zero;             \
        int32_t vote = 0;                                                                        \
        for (uint32_t i = 0; i < n_rows; i++) {                                                  \
            const int8_t *restrict row = rows[i];                                                \
            m0 = add_i8_as_u8(m0, row + c0);                                                     \
            m1 = add_i8_as_u8(m1, row + c1);                                                     \
            if ((N) >= 3) m2 = add_i8_as_u8(m2, row + c2);                                      \
            if ((N) >= 4) m3 = add_i8_as_u8(m3, row + c3);                                      \
            if ((N) >= 5) m4 = add_i8_as_u8(m4, row + c4);                                      \
            if ((N) >= 6) m5 = add_i8_as_u8(m5, row + c5);                                      \
            if ((N) >= 7) m6 = add_i8_as_u8(m6, row + c6);                                      \
            if ((N) >= 8) m7 = add_i8_as_u8(m7, row + c7);                                      \
        }                                                                                        \
        accum_vote_neon(c0, m0, &r, layout->clamp, lane_ids, &vote);                             \
        accum_vote_neon(c1, m1, &r, layout->clamp, lane_ids, &vote);                             \
        if ((N) >= 3) accum_vote_neon(c2, m2, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 4) accum_vote_neon(c3, m3, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 5) accum_vote_neon(c4, m4, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 6) accum_vote_neon(c5, m5, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 7) accum_vote_neon(c6, m6, &r, layout->clamp, lane_ids, &vote);               \
        if ((N) >= 8) accum_vote_neon(c7, m7, &r, layout->clamp, lane_ids, &vote);               \
        return vote;                                                                             \
    }

TM17_NEON_DELTA_N(score_class_delta2_neon, 2)
TM17_NEON_DELTA_N(score_class_delta3_neon, 3)
TM17_NEON_DELTA_N(score_class_delta4_neon, 4)
TM17_NEON_DELTA_N(score_class_delta5_neon, 5)
TM17_NEON_DELTA_N(score_class_delta6_neon, 6)
TM17_NEON_DELTA_N(score_class_delta7_neon, 7)
TM17_NEON_DELTA_N(score_class_delta8_neon, 8)

static int32_t score_class_table_generic_neon(
    const TMLayout *restrict layout,
    const uint8_t *const *restrict rows,
    uint16_t k,
    uint8x16_t zero,
    uint8x16_t lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    int32_t vote = 0;
    uint32_t c = r.pos_start;
    for (; c + 8u * TM17_NEON_LANES <= r.neg_end; c += 8u * TM17_NEON_LANES) {
        const uint32_t c0 = c;
        const uint32_t c1 = c0 + TM17_NEON_LANES;
        const uint32_t c2 = c1 + TM17_NEON_LANES;
        const uint32_t c3 = c2 + TM17_NEON_LANES;
        const uint32_t c4 = c3 + TM17_NEON_LANES;
        const uint32_t c5 = c4 + TM17_NEON_LANES;
        const uint32_t c6 = c5 + TM17_NEON_LANES;
        const uint32_t c7 = c6 + TM17_NEON_LANES;
        uint8x16_t m0 = zero;
        uint8x16_t m1 = zero;
        uint8x16_t m2 = zero;
        uint8x16_t m3 = zero;
        uint8x16_t m4 = zero;
        uint8x16_t m5 = zero;
        uint8x16_t m6 = zero;
        uint8x16_t m7 = zero;
        for (uint32_t b = 0; b < layout->n_bytes; b++) {
            const uint8_t *restrict row = rows[b];
            m0 = vqaddq_u8(m0, vld1q_u8(row + c0));
            m1 = vqaddq_u8(m1, vld1q_u8(row + c1));
            m2 = vqaddq_u8(m2, vld1q_u8(row + c2));
            m3 = vqaddq_u8(m3, vld1q_u8(row + c3));
            m4 = vqaddq_u8(m4, vld1q_u8(row + c4));
            m5 = vqaddq_u8(m5, vld1q_u8(row + c5));
            m6 = vqaddq_u8(m6, vld1q_u8(row + c6));
            m7 = vqaddq_u8(m7, vld1q_u8(row + c7));
        }
        accum_vote_neon(c0, m0, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c1, m1, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c2, m2, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c3, m3, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c4, m4, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c5, m5, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c6, m6, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c7, m7, &r, layout->clamp, lane_ids, &vote);
    }
    for (; c < r.neg_end; c += TM17_NEON_LANES) {
        uint8x16_t mismatches = zero;
        for (uint32_t b = 0; b < layout->n_bytes; b++) {
            mismatches = vqaddq_u8(mismatches, vld1q_u8(rows[b] + c));
        }
        accum_vote_neon(c, mismatches, &r, layout->clamp, lane_ids, &vote);
    }
    return vote;
}

static int32_t score_class_delta_generic_neon(
    const TMLayout *restrict layout,
    const int8_t *const *restrict rows,
    uint32_t n_rows,
    uint16_t k,
    uint8x16_t lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    int32_t vote = 0;
    uint32_t c = r.pos_start;
    for (; c + 8u * TM17_NEON_LANES <= r.neg_end; c += 8u * TM17_NEON_LANES) {
        const uint32_t c0 = c;
        const uint32_t c1 = c0 + TM17_NEON_LANES;
        const uint32_t c2 = c1 + TM17_NEON_LANES;
        const uint32_t c3 = c2 + TM17_NEON_LANES;
        const uint32_t c4 = c3 + TM17_NEON_LANES;
        const uint32_t c5 = c4 + TM17_NEON_LANES;
        const uint32_t c6 = c5 + TM17_NEON_LANES;
        const uint32_t c7 = c6 + TM17_NEON_LANES;
        uint8x16_t m0 = vld1q_u8(layout->base_mismatch_zero + c0);
        uint8x16_t m1 = vld1q_u8(layout->base_mismatch_zero + c1);
        uint8x16_t m2 = vld1q_u8(layout->base_mismatch_zero + c2);
        uint8x16_t m3 = vld1q_u8(layout->base_mismatch_zero + c3);
        uint8x16_t m4 = vld1q_u8(layout->base_mismatch_zero + c4);
        uint8x16_t m5 = vld1q_u8(layout->base_mismatch_zero + c5);
        uint8x16_t m6 = vld1q_u8(layout->base_mismatch_zero + c6);
        uint8x16_t m7 = vld1q_u8(layout->base_mismatch_zero + c7);
        for (uint32_t i = 0; i < n_rows; i++) {
            const int8_t *restrict row = rows[i];
            m0 = add_i8_as_u8(m0, row + c0);
            m1 = add_i8_as_u8(m1, row + c1);
            m2 = add_i8_as_u8(m2, row + c2);
            m3 = add_i8_as_u8(m3, row + c3);
            m4 = add_i8_as_u8(m4, row + c4);
            m5 = add_i8_as_u8(m5, row + c5);
            m6 = add_i8_as_u8(m6, row + c6);
            m7 = add_i8_as_u8(m7, row + c7);
        }
        accum_vote_neon(c0, m0, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c1, m1, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c2, m2, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c3, m3, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c4, m4, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c5, m5, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c6, m6, &r, layout->clamp, lane_ids, &vote);
        accum_vote_neon(c7, m7, &r, layout->clamp, lane_ids, &vote);
    }
    for (; c < r.neg_end; c += TM17_NEON_LANES) {
        uint8x16_t mismatches = vld1q_u8(layout->base_mismatch_zero + c);
        for (uint32_t i = 0; i < n_rows; i++) {
            mismatches = add_i8_as_u8(mismatches, rows[i] + c);
        }
        accum_vote_neon(c, mismatches, &r, layout->clamp, lane_ids, &vote);
    }
    return vote;
}

static int32_t score_class_table_neon(
    const TMLayout *restrict layout,
    const uint8_t *const *restrict rows,
    uint16_t k,
    uint8x16_t zero,
    uint8x16_t lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    const uint32_t n_blocks = (r.neg_end - r.pos_start + TM17_NEON_LANES - 1u) / TM17_NEON_LANES;
    switch (n_blocks) {
        case 2u: return score_class_table2_neon(layout, rows, k, zero, lane_ids);
        case 3u: return score_class_table3_neon(layout, rows, k, zero, lane_ids);
        case 4u: return score_class_table4_neon(layout, rows, k, zero, lane_ids);
        case 5u: return score_class_table5_neon(layout, rows, k, zero, lane_ids);
        case 6u: return score_class_table6_neon(layout, rows, k, zero, lane_ids);
        case 7u: return score_class_table7_neon(layout, rows, k, zero, lane_ids);
        case 8u: return score_class_table8_neon(layout, rows, k, zero, lane_ids);
        default: return score_class_table_generic_neon(layout, rows, k, zero, lane_ids);
    }
}

static int32_t score_class_delta_neon(
    const TMLayout *restrict layout,
    const int8_t *const *restrict rows,
    uint32_t n_rows,
    uint16_t k,
    uint8x16_t zero,
    uint8x16_t lane_ids
) {
    const TM17ClassRange r = class_range(layout, k);
    const uint32_t n_blocks = (r.neg_end - r.pos_start + TM17_NEON_LANES - 1u) / TM17_NEON_LANES;
    switch (n_blocks) {
        case 2u: return score_class_delta2_neon(layout, rows, n_rows, k, zero, lane_ids);
        case 3u: return score_class_delta3_neon(layout, rows, n_rows, k, zero, lane_ids);
        case 4u: return score_class_delta4_neon(layout, rows, n_rows, k, zero, lane_ids);
        case 5u: return score_class_delta5_neon(layout, rows, n_rows, k, zero, lane_ids);
        case 6u: return score_class_delta6_neon(layout, rows, n_rows, k, zero, lane_ids);
        case 7u: return score_class_delta7_neon(layout, rows, n_rows, k, zero, lane_ids);
        case 8u: return score_class_delta8_neon(layout, rows, n_rows, k, zero, lane_ids);
        default: return score_class_delta_generic_neon(layout, rows, n_rows, k, lane_ids);
    }
}

#if defined(TM_ENABLE_AOT_SHAPES)
#define TM17_AOT_NEON_PRED(NAME, NBYTES, NCLASSES, TABLE_FN, DELTA_FN)                         \
    static int32_t NAME(                                                                        \
        const TMLayout *restrict layout,                                                        \
        const uint64_t *restrict current,                                                       \
        int32_t *restrict votes,                                                                \
        uint8x16_t zero,                                                                        \
        uint8x16_t lane_ids                                                                     \
    ) {                                                                                         \
        uint8_t current_bytes[TM_MAX_H_WORDS * 8u];                                             \
        const uint8_t *table_rows[TM_MAX_H_WORDS * 8u];                                         \
        const int8_t *delta_rows[TM_MAX_H_WORDS * 8u];                                          \
        const int use_delta = use_delta_table(layout);                                          \
        uint32_t n_delta = 0u;                                                                  \
        current_to_bytes(current, current_bytes, (NBYTES));                                     \
        if (use_delta) {                                                                        \
            n_delta = select_nonzero_delta_rows(                                                \
                layout->byte_delta_tables,                                                      \
                layout->table_stride,                                                           \
                current_bytes,                                                                  \
                (NBYTES),                                                                       \
                delta_rows                                                                      \
            );                                                                                  \
        } else {                                                                                \
            select_table_rows(layout->byte_tables, layout->table_stride, current_bytes, (NBYTES), table_rows); \
        }                                                                                       \
        int32_t best_vote = INT32_MIN / 2;                                                      \
        int32_t best_class = 0;                                                                 \
        if (use_delta) {                                                                        \
            for (uint16_t k = 0; k < (NCLASSES); k++) {                                         \
                const int32_t vote = DELTA_FN(layout, delta_rows, n_delta, k, zero, lane_ids);   \
                record_vote(k, vote, votes, &best_vote, &best_class);                           \
            }                                                                                   \
        } else {                                                                                \
            for (uint16_t k = 0; k < (NCLASSES); k++) {                                         \
                const int32_t vote = TABLE_FN(layout, table_rows, k, zero, lane_ids);            \
                record_vote(k, vote, votes, &best_vote, &best_class);                           \
            }                                                                                   \
        }                                                                                       \
        return best_class;                                                                      \
    }

TM17_AOT_NEON_PRED(predict_wustl_neon, 31u, 3u, score_class_table4_neon, score_class_delta4_neon)
TM17_AOT_NEON_PRED(predict_nslkdd_neon, 32u, 5u, score_class_table6_neon, score_class_delta6_neon)
TM17_AOT_NEON_PRED(predict_toniot_neon, 16u, 10u, score_class_table7_neon, score_class_delta7_neon)
TM17_AOT_NEON_PRED(predict_medsec_neon, 44u, 5u, score_class_table5_neon, score_class_delta5_neon)
#endif

int32_t tm17_predict_current_neon(
    const TMLayout *restrict layout,
    const uint64_t *restrict current,
    int32_t *restrict votes
) {
    uint8_t current_bytes[TM_MAX_H_WORDS * 8u];
    const uint8_t *table_rows[TM_MAX_H_WORDS * 8u];
    const int8_t *delta_rows[TM_MAX_H_WORDS * 8u];
    static const uint8_t lane_id_bytes[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15
    };
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t lane_ids = vld1q_u8(lane_id_bytes);

#if defined(TM_ENABLE_AOT_SHAPES)
    if (layout->n_classes == 3u && layout->n_bytes == 31u && layout->n_clauses == 180u) {
        return predict_wustl_neon(layout, current, votes, zero, lane_ids);
    }
    if (layout->n_classes == 5u && layout->n_bytes == 32u && layout->n_clauses == 450u) {
        return predict_nslkdd_neon(layout, current, votes, zero, lane_ids);
    }
    if (layout->n_classes == 10u && layout->n_bytes == 16u && layout->n_clauses == 1000u) {
        return predict_toniot_neon(layout, current, votes, zero, lane_ids);
    }
    if (layout->n_classes == 5u && layout->n_bytes == 44u && layout->n_clauses == 400u) {
        return predict_medsec_neon(layout, current, votes, zero, lane_ids);
    }
#endif

    const int use_delta = use_delta_table(layout);
    uint32_t n_delta = 0u;

    if (use_delta) {
        current_to_bytes(current, current_bytes, layout->n_bytes);
        n_delta = select_nonzero_delta_rows(
            layout->byte_delta_tables,
            layout->table_stride,
            current_bytes,
            layout->n_bytes,
            delta_rows
        );
    } else {
        current_to_bytes(current, current_bytes, layout->n_bytes);
        select_table_rows(
            layout->byte_tables,
            layout->table_stride,
            current_bytes,
            layout->n_bytes,
            table_rows
        );
    }

    int32_t best_vote = INT32_MIN / 2;
    int32_t best_class = 0;
    for (uint16_t k = 0; k < layout->n_classes; k++) {
        const int32_t vote = use_delta
            ? score_class_delta_neon(layout, delta_rows, n_delta, k, zero, lane_ids)
            : score_class_table_neon(layout, table_rows, k, zero, lane_ids);
        record_vote(k, vote, votes, &best_vote, &best_class);
    }
    return best_class;
}
#endif
