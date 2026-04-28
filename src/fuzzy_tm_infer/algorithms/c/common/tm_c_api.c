#include "tm_common.h"

#include <stdint.h>
#include <stdlib.h>

typedef struct {
    FBZModel model;
    TMLayout layout;
    void *scratch;
    int32_t *votes;
} FuzzyTMHandle;

void *fuzzy_tm_model_load(const char *model_path) {
    if (model_path == NULL) {
        return NULL;
    }

    FuzzyTMHandle *model = xmalloc(sizeof(*model));
    model->model = read_fbz(model_path);
    model->layout = tm_adapter_build_layout(&model->model);
    model->scratch = tm_adapter_create_scratch(&model->layout);
    model->votes = xaligned_alloc(64, (size_t)model->layout.n_classes * sizeof(int32_t));
    return model;
}

void fuzzy_tm_model_free(void *handle) {
    FuzzyTMHandle *model = (FuzzyTMHandle *)handle;
    if (model == NULL) {
        return;
    }
    free(model->votes);
    tm_adapter_free_scratch(model->scratch);
    tm_adapter_free_layout(&model->layout);
    free_fbz(&model->model);
    free(model);
}

uint16_t fuzzy_tm_n_literals(const void *handle) {
    const FuzzyTMHandle *model = (const FuzzyTMHandle *)handle;
    return model != NULL ? model->layout.n_literals : 0u;
}

uint16_t fuzzy_tm_n_classes(const void *handle) {
    const FuzzyTMHandle *model = (const FuzzyTMHandle *)handle;
    return model != NULL ? model->layout.n_classes : 0u;
}

uint32_t fuzzy_tm_h_words(const void *handle) {
    const FuzzyTMHandle *model = (const FuzzyTMHandle *)handle;
    return model != NULL ? model->layout.h_words : 0u;
}

void fuzzy_tm_model_calibrate(
    void *handle,
    const float *rows,
    uint32_t n_rows,
    uint32_t n_features,
    int verbose
) {
    FuzzyTMHandle *model = (FuzzyTMHandle *)handle;
    if (model == NULL || rows == NULL || n_rows == 0u || n_features == 0u) {
        return;
    }
    Matrix x;
    x.n_rows = n_rows;
    x.n_features = n_features;
    x.x = (float *)rows;
    tm_adapter_calibrate_layout(&x, &model->layout, verbose);
}

int32_t fuzzy_tm_predict_row(void *handle, const float *row) {
    FuzzyTMHandle *model = (FuzzyTMHandle *)handle;
    if (model == NULL || row == NULL) {
        return -1;
    }
    return tm_adapter_predict_row(&model->layout, row, model->scratch, model->votes);
}

void fuzzy_tm_predict_batch(
    void *handle,
    const float *rows,
    uint32_t n_rows,
    uint32_t row_stride,
    int32_t *out_classes
) {
    FuzzyTMHandle *model = (FuzzyTMHandle *)handle;
    if (model == NULL || rows == NULL || out_classes == NULL) {
        return;
    }
    for (uint32_t i = 0; i < n_rows; i++) {
        out_classes[i] = tm_adapter_predict_row(
            &model->layout,
            rows + (size_t)i * row_stride,
            model->scratch,
            model->votes
        );
    }
}

/* Backward-compatible ABI aliases for existing local experiments. */
void *tm_c_model_load(const char *model_path) { return fuzzy_tm_model_load(model_path); }
void tm_c_model_free(void *handle) { fuzzy_tm_model_free(handle); }
uint16_t tm_c_n_literals(const void *handle) { return fuzzy_tm_n_literals(handle); }
uint16_t tm_c_n_classes(const void *handle) { return fuzzy_tm_n_classes(handle); }
uint32_t tm_c_h_words(const void *handle) { return fuzzy_tm_h_words(handle); }
void tm_c_model_calibrate(void *handle, const float *rows, uint32_t n_rows, uint32_t n_features, int verbose) {
    fuzzy_tm_model_calibrate(handle, rows, n_rows, n_features, verbose);
}
int32_t tm_c_predict_row(void *handle, const float *row) { return fuzzy_tm_predict_row(handle, row); }
void tm_c_predict_batch(void *handle, const float *rows, uint32_t n_rows, uint32_t row_stride, int32_t *out_classes) {
    fuzzy_tm_predict_batch(handle, rows, n_rows, row_stride, out_classes);
}
