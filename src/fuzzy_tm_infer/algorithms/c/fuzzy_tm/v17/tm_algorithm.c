#include "tm_algorithm.h"
#include "tm_kernel_17.h"

/*
 * Split-kernel candidate.
 *
 * This file owns binarization and public API glue. Architecture-specific
 * scoring lives in tm_kernel_17.c so the hot AVX2/NEON loops stay isolated.
 */

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__) && !defined(TM_NO_AVX2_BINARIZE)
#include <immintrin.h>
#endif

#define TM_AVX2_LANES 32u
#define TM_NEON_LANES 16u

static inline void set_literal_range(uint64_t *restrict current, uint32_t begin, uint32_t count) {
    if (count == 0u) {
        return;
    }

    const uint32_t end = begin + count;
    const uint32_t w0 = begin >> 6;
    const uint32_t w1 = (end - 1u) >> 6;
    const uint32_t b0 = begin & 63u;
    const uint32_t b1 = (end - 1u) & 63u;

    if (w0 == w1) {
        current[w0] |= (~UINT64_C(0) << b0) & (~UINT64_C(0) >> (63u - b1));
        return;
    }

    current[w0] |= ~UINT64_C(0) << b0;
    for (uint32_t w = w0 + 1u; w < w1; w++) {
        current[w] = ~UINT64_C(0);
    }
    current[w1] |= ~UINT64_C(0) >> (63u - b1);
}

static void binarize_feature_blocks(
    const TMLayout *restrict layout,
    const float *restrict row,
    uint64_t *restrict current
) {
    memset(current, 0, (size_t)layout->h_words * sizeof(uint64_t));

    for (uint32_t fb = 0; fb < layout->n_feature_blocks; fb++) {
        const TMFeatureBlock *restrict block = layout->feature_blocks + fb;
        const float value = row[block->feature];
        const float *restrict thresh = layout->thresh + block->literal_begin;
        uint32_t lo = 0;
        uint32_t hi = block->count;

        if (hi <= 8u) {
            while (lo < hi && value >= thresh[lo]) {
                lo++;
            }
        } else {
            while (lo < hi) {
                const uint32_t mid = lo + ((hi - lo) >> 1);
                if (value >= thresh[mid]) {
                    lo = mid + 1u;
                } else {
                    hi = mid;
                }
            }
        }

        set_literal_range(current, block->literal_begin, lo);
    }
}

void tm_binarize_row(
    const TMLayout *restrict layout,
    const float *restrict row,
    uint64_t *restrict current
) {
    if (layout->prefer_feature_blocks != 0u && layout->feature_blocks != NULL) {
        binarize_feature_blocks(layout, row, current);
        return;
    }

    uint16_t i = 0;
    for (uint32_t h = 0; h < layout->h_words; h++) {
        uint64_t word = 0;
        uint16_t end = (uint16_t)((h + 1u) << 6);
        if (end > layout->n_literals) {
            end = layout->n_literals;
        }

#if defined(__AVX2__) && !defined(TM_NO_AVX2_BINARIZE)
        for (; (uint16_t)(i + 8u) <= end; i = (uint16_t)(i + 8u)) {
            const __m256 values = _mm256_i32gather_ps(
                row,
                _mm256_loadu_si256((const __m256i *)(layout->feat_idx + i)),
                4
            );
            const __m256 thresh = _mm256_loadu_ps(layout->thresh + i);
            const __m256 cmp = _mm256_cmp_ps(values, thresh, _CMP_GE_OQ);
            word |= (uint64_t)_mm256_movemask_ps(cmp) << (i & 63);
        }
#endif

        if (layout->feat_idx_u16 != NULL) {
            const uint16_t *restrict feat = layout->feat_idx_u16;
            for (; i < end; i++) {
                word |= (uint64_t)(row[feat[i]] >= layout->thresh[i]) << (i & 63);
            }
        } else {
            for (; i < end; i++) {
                word |= (uint64_t)(row[layout->feat_idx[i]] >= layout->thresh[i]) << (i & 63);
            }
        }
        current[h] = word;
    }
}

int32_t tm_predict_current(
    const TMLayout *restrict layout,
    const uint64_t *restrict current,
    int32_t *restrict votes
) {
#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
    if (layout->byte_tables != NULL && layout->table_stride >= layout->n_clauses + TM_AVX2_LANES - 1u) {
        return tm17_predict_current_avx2(layout, current, votes);
    }
#endif

#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
    if (layout->byte_tables != NULL && layout->table_stride >= layout->n_clauses + TM_NEON_LANES - 1u) {
        return tm17_predict_current_neon(layout, current, votes);
    }
#endif

    (void)layout;
    (void)current;
    (void)votes;
    return 0;
}

int32_t tm_predict_row(
    const TMLayout *restrict layout,
    const float *restrict row,
    uint64_t *restrict current,
    int32_t *restrict votes
) {
    tm_binarize_row(layout, row, current);
    return tm_predict_current(layout, current, votes);
}
