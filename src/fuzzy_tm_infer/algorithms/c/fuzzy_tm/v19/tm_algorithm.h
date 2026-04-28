#ifndef TM_ALGORITHM_H
#define TM_ALGORITHM_H

#include <stdint.h>

#define TM_MAX_LITERALS 2048u
#define TM_MAX_H_WORDS (TM_MAX_LITERALS / 64u)

typedef struct {
    uint32_t feature;
    uint16_t literal_begin;
    uint16_t count;
} TMFeatureBlock;

#define TM_MAX_SCORE_BLOCK_LANES 32u

typedef struct {
    uint32_t start;
    uint16_t class_id;
    uint8_t lanes;
    uint8_t pos_lanes;
    uint8_t clamp[TM_MAX_SCORE_BLOCK_LANES];
    uint8_t base[TM_MAX_SCORE_BLOCK_LANES];
} TMScoreBlock;

typedef struct {
    uint16_t n_literals;
    uint16_t n_classes;
    uint32_t h_words;
    uint32_t n_clauses;

    /* Feature-threshold state layout. */
    float *thresh;
    TMFeatureBlock *feature_blocks;
    uint32_t n_feature_blocks;

    /* Temporary build-time dense clause layout. Freed after feature-state tables are built. */
    uint64_t *inter;
    uint8_t *base_mismatch_zero;

    /* Clause metadata, grouped per class as positive then negative ranges. */
    uint8_t *clamp;
    int32_t *pos_start;
    int32_t *pos_end;
    int32_t *neg_start;
    int32_t *neg_end;

    /* Feature-state score blocks: votes += +/-sum(ReLU(clamp - mismatch)). */
    uint32_t score_block_lanes;
    uint32_t n_score_blocks;
    TMScoreBlock *score_blocks;

    uint32_t n_feature_state_values;
    uint32_t *feature_state_offsets;
    int8_t *blocked_feature_state_delta_tables;
} TMLayout;

int32_t tm_predict_row(
    const TMLayout *restrict layout,
    const float *restrict row,
    int32_t *restrict votes
);

#endif
