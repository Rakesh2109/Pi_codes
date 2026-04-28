#include "tm_algorithm.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#if !defined(TM_ENABLE_SCORE_BLOCKS) || !defined(TM_ENABLE_FEATURE_STATE_TABLES)
#error "tm_v19 requires -DTM_ENABLE_SCORE_BLOCKS -DTM_ENABLE_FEATURE_STATE_TABLES"
#endif

#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
#include <arm_neon.h>
#endif

#if !(defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)) && \
    !(defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE))
#error "tm_v19 requires AVX2 or NEON scoring"
#endif

#define TM_AVX2_LANES 32u
#define TM_NEON_LANES 16u

static inline uint8_t feature_state(
    const TMLayout *restrict l,
    const float *restrict row,
    uint32_t fb
) {
    const TMFeatureBlock *b = l->feature_blocks + fb;
    const float v = row[b->feature];
    const float *t = l->thresh + b->literal_begin;
    uint32_t lo = 0, hi = b->count;
    if (hi <= 8u) {
        while (lo < hi && v >= t[lo]) lo++;
    } else {
        while (lo < hi) {
            const uint32_t mid = lo + ((hi - lo) >> 1);
            if (v >= t[mid]) lo = mid + 1u; else hi = mid;
        }
    }
    return (uint8_t)lo;
}

static inline void feature_states(
    const TMLayout *restrict l,
    const float *restrict row,
    uint8_t *restrict states
) {
    for (uint32_t fb = 0; fb < l->n_feature_blocks; fb++) {
        states[fb] = feature_state(l, row, fb);
    }
}

#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
static inline int hsum32(__m256i v) {
    const __m256i z = _mm256_setzero_si256();
    const __m256i s = _mm256_sad_epu8(v, z);
    const __m128i lo = _mm256_castsi256_si128(s), hi = _mm256_extracti128_si256(s, 1);
    const __m128i q = _mm_add_epi64(lo, hi);
    return (int)((uint64_t)_mm_cvtsi128_si64(q) + (uint64_t)_mm_extract_epi64(q, 1));
}

static int32_t predict_states_avx2(const TMLayout *restrict l, const uint8_t *restrict states, int32_t *restrict votes) {
    const __m256i lane_ids = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    );
    memset(votes, 0, (size_t)l->n_classes * sizeof(int32_t));

    for (uint32_t bi = 0; bi < l->n_score_blocks; bi++) {
        const TMScoreBlock *blk = l->score_blocks + bi;
        __m256i m = _mm256_loadu_si256((const __m256i *)blk->base);
        for (uint32_t fb = 0; fb < l->n_feature_blocks; fb++) {
            const uint32_t state = l->feature_state_offsets[fb] + states[fb];
            const int8_t *r =
                l->blocked_feature_state_delta_tables +
                ((size_t)bi * l->n_feature_state_values + state) * TM_AVX2_LANES;
            m = _mm256_add_epi8(m, _mm256_load_si256((const __m256i *)r));
        }
        __m256i s = _mm256_subs_epu8(_mm256_loadu_si256((const __m256i *)blk->clamp), m);
        if (blk->lanes < TM_AVX2_LANES) {
            s = _mm256_and_si256(s, _mm256_cmpgt_epi8(_mm256_set1_epi8((char)blk->lanes), lane_ids));
        }
        const int total = hsum32(s);
        if (blk->pos_lanes == 0u) votes[blk->class_id] -= total;
        else if (blk->pos_lanes >= blk->lanes) votes[blk->class_id] += total;
        else {
            const __m256i mask = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)blk->pos_lanes), lane_ids);
            votes[blk->class_id] += 2 * hsum32(_mm256_and_si256(s, mask)) - total;
        }
    }

    int32_t best = 0, bestv = INT32_MIN / 2;
    for (uint16_t k = 0; k < l->n_classes; k++) if (votes[k] > bestv) { bestv = votes[k]; best = k; }
    return best;
}
#endif

#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
static inline int hsum16(uint8x16_t v) {
#if defined(__aarch64__)
    return (int)vaddlvq_u8(v);
#else
    const uint16x8_t s16 = vpaddlq_u8(v);
    const uint32x4_t s32 = vpaddlq_u16(s16);
    const uint64x2_t s64 = vpaddlq_u32(s32);
    return (int)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));
#endif
}

static int32_t predict_states_neon(const TMLayout *restrict l, const uint8_t *restrict states, int32_t *restrict votes) {
    static const uint8_t lane_id_bytes[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const uint8x16_t lane_ids = vld1q_u8(lane_id_bytes);
    memset(votes, 0, (size_t)l->n_classes * sizeof(int32_t));

    for (uint32_t bi = 0; bi < l->n_score_blocks; bi++) {
        const TMScoreBlock *blk = l->score_blocks + bi;
        uint8x16_t m = vld1q_u8(blk->base);
        for (uint32_t fb = 0; fb < l->n_feature_blocks; fb++) {
            const uint32_t state = l->feature_state_offsets[fb] + states[fb];
            const int8_t *r =
                l->blocked_feature_state_delta_tables +
                ((size_t)bi * l->n_feature_state_values + state) * TM_NEON_LANES;
            m = vreinterpretq_u8_s8(vaddq_s8(vreinterpretq_s8_u8(m), vld1q_s8(r)));
        }
        uint8x16_t s = vqsubq_u8(vld1q_u8(blk->clamp), m);
        if (blk->lanes < TM_NEON_LANES) s = vandq_u8(s, vcltq_u8(lane_ids, vdupq_n_u8(blk->lanes)));
        const int total = hsum16(s);
        if (blk->pos_lanes == 0u) votes[blk->class_id] -= total;
        else if (blk->pos_lanes >= blk->lanes) votes[blk->class_id] += total;
        else {
            const uint8x16_t mask = vcltq_u8(lane_ids, vdupq_n_u8(blk->pos_lanes));
            votes[blk->class_id] += 2 * hsum16(vandq_u8(s, mask)) - total;
        }
    }

    int32_t best = 0, bestv = INT32_MIN / 2;
    for (uint16_t k = 0; k < l->n_classes; k++) if (votes[k] > bestv) { bestv = votes[k]; best = k; }
    return best;
}
#endif

int32_t tm_predict_row(const TMLayout *restrict l, const float *restrict row, int32_t *restrict votes) {
    uint8_t states[TM_MAX_LITERALS];
    feature_states(l, row, states);
#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
    if (l->score_block_lanes == TM_AVX2_LANES && l->blocked_feature_state_delta_tables) {
        return predict_states_avx2(l, states, votes);
    }
#endif
#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
    if (l->score_block_lanes == TM_NEON_LANES && l->blocked_feature_state_delta_tables) {
        return predict_states_neon(l, states, votes);
    }
#endif
    __builtin_unreachable();
}
