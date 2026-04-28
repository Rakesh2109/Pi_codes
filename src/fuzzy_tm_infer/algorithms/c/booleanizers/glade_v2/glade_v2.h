#ifndef FUZZY_TM_GLADE_V2_H
#define FUZZY_TM_GLADE_V2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *glade_v2_backend(void);

uint32_t glade_v2_profile_field_count(void);

const char *glade_v2_profile_field_name(uint32_t index);

int glade_v2_profile_enabled(void);

void glade_v2_profile_reset(void);

int glade_v2_profile_read(uint64_t *out, uint32_t capacity);

int glade_v2_transform_u8(
    const double *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t n_bits,
    uint8_t *out
);

int glade_v2_transform_u8_f32(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
);

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
);

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
);

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
);

int glade_v2_transform_packed(
    const double *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const double *thresholds,
    uint32_t n_bits,
    uint8_t *out
);

int glade_v2_transform_packed_f32(
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t *feature_indices,
    const float *thresholds,
    uint32_t n_bits,
    uint8_t *out
);

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
);

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
);

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
);

#ifdef __cplusplus
}
#endif

#endif
