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

typedef struct {
    uint16_t n_literals;
    uint16_t n_classes;
    uint32_t h_words;
    uint32_t n_clauses;

    /* Derived layout sizes. Strides are padded for vector tail loads. */
    uint32_t n_bytes;
    uint32_t table_stride;

    /* Set by the harness calibration pass; algorithm code only consumes it. */
    uint8_t prefer_feature_blocks;
    uint8_t prefer_delta_table;
    uint8_t delta_table_exact;

    /* Binarization layout. */
    int32_t *feat_idx;
    uint16_t *feat_idx_u16;
    float *thresh;
    TMFeatureBlock *feature_blocks;
    uint32_t n_feature_blocks;

    /* Dense build layout and byte-table scorers. */
    uint64_t *inter;
    uint8_t *byte_tables;
    int8_t *byte_delta_tables;
    uint8_t *base_mismatch_zero;

    /* Clause metadata, grouped per class as positive then negative ranges. */
    uint8_t *clamp;
    int32_t *pos_start;
    int32_t *pos_end;
    int32_t *neg_start;
    int32_t *neg_end;
} TMLayout;

void tm_binarize_row(
    const TMLayout *restrict layout,
    const float *restrict row,
    uint64_t *restrict current
);

int32_t tm_predict_current(
    const TMLayout *restrict layout,
    const uint64_t *restrict current,
    int32_t *restrict votes
);

int32_t tm_predict_row(
    const TMLayout *restrict layout,
    const float *restrict row,
    uint64_t *restrict current,
    int32_t *restrict votes
);

#if defined(TM_ENABLE_BATCH_API)
void tm_predict_batch(
    const TMLayout *restrict layout,
    const float *restrict rows,
    uint32_t n_rows,
    uint32_t row_stride,
    uint64_t *restrict currents,
    int32_t *restrict votes,
    int32_t *restrict out_classes
);
#endif

#endif
