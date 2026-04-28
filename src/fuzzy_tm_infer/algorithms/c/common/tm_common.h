#ifndef TM_COMMON_H
#define TM_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "tm_algorithm.h"

typedef struct {
    const char *stem;
    const char *name;
    const char *test_dir;
} Dataset;

typedef struct {
    uint16_t n_literals;
    uint16_t n_classes;
    uint32_t h_words;
    uint32_t n_clauses;
    int32_t *feat_idx;
    float *thresh;
    uint64_t *lits;
    uint64_t *inv;
    int32_t *clamp;
    int32_t *sign;
    int32_t *cls;
} FBZModel;

typedef struct {
    uint32_t n_rows;
    uint32_t n_features;
    float *x;
} Matrix;

void *xmalloc(size_t n);
void *xcalloc(size_t count, size_t size);
void *xaligned_alloc(size_t alignment, size_t n);
double now_us(void);

uint16_t read_u16_le(const uint8_t *p);
uint32_t read_u32_le(const uint8_t *p);
int32_t read_i32_le(const uint8_t *p);
float read_f32_le(const uint8_t *p);
uint64_t read_u64_from_le_bytes(const uint8_t *p, uint32_t available);
uint32_t popcount64_u32(uint64_t v);
uint64_t valid_literal_mask(uint32_t word_index, uint16_t n_literals);
uint32_t ceil_div_u32(uint32_t n, uint32_t d);

uint8_t *read_file(const char *path, size_t *out_size);
FBZModel read_fbz(const char *path);
Matrix load_x(const char *path);
int32_t *load_y(const char *path, uint32_t n_rows);
double macro_f1(const int32_t *y_true, const int32_t *y_pred, uint32_t n, uint16_t k);
void free_fbz(FBZModel *m);
void write_predictions(const char *pred_dir, const char *stem, const int32_t *y_pred, uint32_t n_rows);

TMLayout tm_adapter_build_layout(const FBZModel *model);
void tm_adapter_calibrate_layout(const Matrix *x, TMLayout *layout, int verbose);
void tm_adapter_free_layout(TMLayout *layout);
void *tm_adapter_create_scratch(const TMLayout *layout);
void tm_adapter_free_scratch(void *scratch);
int32_t tm_adapter_predict_row(
    const TMLayout *layout,
    const float *row,
    void *scratch,
    int32_t *votes
);
void tm_adapter_print_profile(const TMLayout *layout, const Matrix *x, void *scratch, int32_t *votes, int detail);
void tm_adapter_print_stats(const FBZModel *model, const TMLayout *layout);
const char *tm_adapter_profile_label(void);
const char *tm_adapter_usage_suffix(void);

#endif
