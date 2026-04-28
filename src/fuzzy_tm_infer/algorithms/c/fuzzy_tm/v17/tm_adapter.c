#include "../../common/tm_common.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TM_CALIBRATION_MAX_ROWS 3000u
#define TM_CALIBRATION_REPEATS 3u
#define TM_CALIBRATION_TRIALS 3u
#define TM_CALIBRATION_WARMUP_ROWS 128u
#define TM_BINARIZE_SELECT_MARGIN 0.985
#define TM_DELTA_SELECT_MARGIN 0.985
#define TM_DELTA_DENSE_BYTES_NUMERATOR 3u
#define TM_DELTA_DENSE_BYTES_DENOMINATOR 4u
#define TM_PROFILE_MAX_ROWS 3000u

#if !defined(TM_RANGE_ALIGN_LANES)
#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
#define TM_RANGE_ALIGN_LANES 32u
#elif defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
#define TM_RANGE_ALIGN_LANES 16u
#else
#define TM_RANGE_ALIGN_LANES 1u
#endif
#endif

typedef struct {
    const FBZModel *model;
    uint32_t idx;
} ClauseOrder;

typedef struct {
    uint64_t *current;
} TMAdapterScratch;

typedef struct {
    double avg_active_literals_per_clause;
    double avg_active_words_per_clause;
    uint32_t empty_clauses;
    uint32_t linearizable_clauses;
    uint32_t clamp1_clauses;
    uint32_t unused_literals;
    uint64_t total_postings;
    uint64_t dense_bytes;
    uint64_t byte_table_bytes;
    uint64_t delta_table_bytes;
    uint64_t base_mismatch_bytes;
} TMModelStats;

static uint8_t popcount8(uint8_t v) {
    v = (uint8_t)(v - ((v >> 1u) & 0x55u));
    v = (uint8_t)((v & 0x33u) + ((v >> 2u) & 0x33u));
    return (uint8_t)((v + (v >> 4u)) & 0x0fu);
}

static uint32_t round_up_u32(uint32_t n, uint32_t align) {
    return align <= 1u ? n : ((n + align - 1u) / align) * align;
}

static int should_lane_align_ranges(uint32_t pos_count, uint32_t neg_count, uint32_t lanes) {
    const uint32_t mixed_blocks = ceil_div_u32(pos_count + neg_count, lanes);
    const uint32_t aligned_blocks = ceil_div_u32(pos_count, lanes) + ceil_div_u32(neg_count, lanes);
    return aligned_blocks <= mixed_blocks;
}

static uint32_t maybe_aligned_span(uint32_t count, int align_ranges, uint32_t lanes) {
    return align_ranges ? round_up_u32(count, lanes) : count;
}

static int clause_key(const FBZModel *m, uint32_t idx) {
    return m->cls[idx] * 4 + (1 - (m->sign[idx] > 0 ? 1 : 0));
}

static int cmp_clause_order(const void *a, const void *b) {
    const ClauseOrder *oa = (const ClauseOrder *)a;
    const ClauseOrder *ob = (const ClauseOrder *)b;
    int ka = clause_key(oa->model, oa->idx);
    int kb = clause_key(ob->model, ob->idx);
    if (ka != kb) {
        return ka < kb ? -1 : 1;
    }
    return oa->idx < ob->idx ? -1 : (oa->idx > ob->idx);
}

static TMFeatureBlock *build_feature_blocks(const FBZModel *m, uint32_t *out_n_feature_blocks) {
    *out_n_feature_blocks = 0;
    if (m->n_literals == 0) {
        return NULL;
    }
    TMFeatureBlock *blocks = xmalloc((size_t)m->n_literals * sizeof(TMFeatureBlock));
    uint32_t n_blocks = 0;
    uint16_t begin = 0;
    while (begin < m->n_literals) {
        const int32_t feature = m->feat_idx[begin];
        if (feature < 0) {
            free(blocks);
            return NULL;
        }
        uint16_t end = (uint16_t)(begin + 1u);
        while (end < m->n_literals && m->feat_idx[end] == feature) {
            if (m->thresh[end - 1u] > m->thresh[end]) {
                free(blocks);
                return NULL;
            }
            end++;
        }
        blocks[n_blocks].feature = (uint32_t)feature;
        blocks[n_blocks].literal_begin = begin;
        blocks[n_blocks].count = (uint16_t)(end - begin);
        n_blocks++;
        begin = end;
    }
    *out_n_feature_blocks = n_blocks;
    return blocks;
}

static void write_layout_clause(TMLayout *l, const FBZModel *m, uint32_t out, uint32_t in, int32_t *sorted_cls, int32_t *sorted_sign) {
    l->clamp[out] = (uint8_t)m->clamp[in];
    sorted_cls[out] = m->cls[in];
    sorted_sign[out] = m->sign[in];
    for (uint32_t h = 0; h < l->h_words; h++) {
        uint64_t lits = m->lits[(size_t)in * l->h_words + h];
        uint64_t inv = m->inv[(size_t)in * l->h_words + h];
        l->inter[(size_t)out * 2u * l->h_words + 2u * h] = lits;
        l->inter[(size_t)out * 2u * l->h_words + 2u * h + 1u] = lits ^ inv;
    }

    uint32_t base_mismatches_zero = 0;
    uint32_t max_mismatches = 0;
    for (uint32_t b = 0; b < l->n_bytes; b++) {
        const uint32_t h = b >> 3;
        const uint32_t shift = 8u * (b & 7u);
        const uint8_t lits_byte = (uint8_t)(l->inter[(size_t)out * 2u * l->h_words + 2u * h] >> shift);
        const uint8_t xor_byte = (uint8_t)(l->inter[(size_t)out * 2u * l->h_words + 2u * h + 1u] >> shift);
        const uint8_t base = popcount8(lits_byte);
        base_mismatches_zero += base;
        max_mismatches += popcount8((uint8_t)(xor_byte | lits_byte));
        for (uint32_t v = 0; v < 256u; v++) {
            const uint8_t contrib = popcount8((uint8_t)(lits_byte ^ (xor_byte & (uint8_t)v)));
            l->byte_tables[((size_t)b * 256u + v) * l->table_stride + out] = contrib;
            l->byte_delta_tables[((size_t)b * 256u + v) * l->table_stride + out] =
                (int8_t)((int32_t)contrib - (int32_t)base);
        }
    }
    if (max_mismatches > UINT8_MAX) {
        l->delta_table_exact = 0u;
    }
    l->base_mismatch_zero[out] = (uint8_t)base_mismatches_zero;
}

static void write_dummy_clause(TMLayout *l, uint32_t out, uint16_t cls, int32_t sign, int32_t *sorted_cls, int32_t *sorted_sign) {
    sorted_cls[out] = cls;
    sorted_sign[out] = sign;
    memset(l->inter + (size_t)out * 2u * l->h_words, 0, (size_t)2u * l->h_words * sizeof(uint64_t));
}

TMLayout tm_adapter_build_layout(const FBZModel *m) {
    TMLayout l;
    memset(&l, 0, sizeof(l));
    l.n_literals = m->n_literals;
    l.n_classes = m->n_classes;
    l.h_words = m->h_words;

    uint32_t *pos_counts = xcalloc(l.n_classes, sizeof(uint32_t));
    uint32_t *neg_counts = xcalloc(l.n_classes, sizeof(uint32_t));
    for (uint32_t i = 0; i < m->n_clauses; i++) {
        if (m->sign[i] > 0) {
            pos_counts[m->cls[i]]++;
        } else {
            neg_counts[m->cls[i]]++;
        }
    }

    for (uint16_t k = 0; k < l.n_classes; k++) {
        const int align_ranges = should_lane_align_ranges(pos_counts[k], neg_counts[k], TM_RANGE_ALIGN_LANES);
        l.n_clauses += maybe_aligned_span(pos_counts[k], align_ranges, TM_RANGE_ALIGN_LANES);
        l.n_clauses += maybe_aligned_span(neg_counts[k], align_ranges, TM_RANGE_ALIGN_LANES);
    }

    l.n_bytes = (l.n_literals + 7u) / 8u;
    l.table_stride = ((l.n_clauses + 31u) & ~31u) + 32u;
    l.delta_table_exact = 1u;
    l.feat_idx = xmalloc((size_t)l.n_literals * sizeof(int32_t));
    l.feat_idx_u16 = xmalloc((size_t)l.n_literals * sizeof(uint16_t));
    l.thresh = xmalloc((size_t)l.n_literals * sizeof(float));
    memcpy(l.feat_idx, m->feat_idx, (size_t)l.n_literals * sizeof(int32_t));
    for (uint16_t i = 0; i < l.n_literals; i++) {
        if (m->feat_idx[i] < 0 || m->feat_idx[i] > UINT16_MAX) {
            free(l.feat_idx_u16);
            l.feat_idx_u16 = NULL;
            break;
        }
        l.feat_idx_u16[i] = (uint16_t)m->feat_idx[i];
    }
    memcpy(l.thresh, m->thresh, (size_t)l.n_literals * sizeof(float));
    l.feature_blocks = build_feature_blocks(m, &l.n_feature_blocks);

    l.inter = xaligned_alloc(64, (size_t)l.n_clauses * 2u * l.h_words * sizeof(uint64_t));
    l.byte_tables = xaligned_alloc(64, (size_t)l.n_bytes * 256u * l.table_stride * sizeof(uint8_t));
    memset(l.byte_tables, 0, (size_t)l.n_bytes * 256u * l.table_stride * sizeof(uint8_t));
    l.byte_delta_tables = xaligned_alloc(64, (size_t)l.n_bytes * 256u * l.table_stride * sizeof(int8_t));
    memset(l.byte_delta_tables, 0, (size_t)l.n_bytes * 256u * l.table_stride * sizeof(int8_t));
    l.base_mismatch_zero = xcalloc(l.table_stride, sizeof(uint8_t));
    l.clamp = xcalloc(l.table_stride, sizeof(uint8_t));
    l.pos_start = xcalloc(l.n_classes, sizeof(int32_t));
    l.pos_end = xcalloc(l.n_classes, sizeof(int32_t));
    l.neg_start = xcalloc(l.n_classes, sizeof(int32_t));
    l.neg_end = xcalloc(l.n_classes, sizeof(int32_t));

    ClauseOrder *order = xmalloc((size_t)m->n_clauses * sizeof(ClauseOrder));
    for (uint32_t i = 0; i < m->n_clauses; i++) {
        order[i].model = m;
        order[i].idx = i;
    }
    qsort(order, m->n_clauses, sizeof(ClauseOrder), cmp_clause_order);
    int32_t *sorted_cls = xmalloc((size_t)l.n_clauses * sizeof(int32_t));
    int32_t *sorted_sign = xmalloc((size_t)l.n_clauses * sizeof(int32_t));

    uint32_t out = 0;
    for (uint16_t k = 0; k < l.n_classes; k++) {
        const uint32_t pos_begin = out;
        for (uint32_t ord = 0; ord < m->n_clauses; ord++) {
            const uint32_t in = order[ord].idx;
            if (m->cls[in] == k && m->sign[in] > 0) {
                write_layout_clause(&l, m, out++, in, sorted_cls, sorted_sign);
            }
        }
        const int align_ranges = should_lane_align_ranges(pos_counts[k], neg_counts[k], TM_RANGE_ALIGN_LANES);
        const uint32_t pos_padded_end = pos_begin + maybe_aligned_span(pos_counts[k], align_ranges, TM_RANGE_ALIGN_LANES);
        while (out < pos_padded_end) {
            write_dummy_clause(&l, out++, k, 1, sorted_cls, sorted_sign);
        }
        const uint32_t neg_begin = out;
        for (uint32_t ord = 0; ord < m->n_clauses; ord++) {
            const uint32_t in = order[ord].idx;
            if (m->cls[in] == k && m->sign[in] < 0) {
                write_layout_clause(&l, m, out++, in, sorted_cls, sorted_sign);
            }
        }
        const uint32_t neg_padded_end = neg_begin + maybe_aligned_span(neg_counts[k], align_ranges, TM_RANGE_ALIGN_LANES);
        while (out < neg_padded_end) {
            write_dummy_clause(&l, out++, k, -1, sorted_cls, sorted_sign);
        }
    }

    for (uint16_t k = 0; k < l.n_classes; k++) {
        for (uint32_t i = 0; i < l.n_clauses; i++) {
            if (sorted_cls[i] == k && sorted_sign[i] > 0) {
                if (l.pos_end[k] == 0) l.pos_start[k] = (int32_t)i;
                l.pos_end[k] = (int32_t)i + 1;
            }
            if (sorted_cls[i] == k && sorted_sign[i] < 0) {
                if (l.neg_end[k] == 0) l.neg_start[k] = (int32_t)i;
                l.neg_end[k] = (int32_t)i + 1;
            }
        }
    }

    free(sorted_cls);
    free(sorted_sign);
    free(order);
    free(pos_counts);
    free(neg_counts);
    return l;
}

void tm_adapter_free_layout(TMLayout *l) {
    free(l->feat_idx);
    free(l->feat_idx_u16);
    free(l->thresh);
    free(l->feature_blocks);
    free(l->inter);
    free(l->byte_tables);
    free(l->byte_delta_tables);
    free(l->base_mismatch_zero);
    free(l->clamp);
    free(l->pos_start);
    free(l->pos_end);
    free(l->neg_start);
    free(l->neg_end);
}

void *tm_adapter_create_scratch(const TMLayout *layout) {
    TMAdapterScratch *scratch = xmalloc(sizeof(*scratch));
    scratch->current = xaligned_alloc(64, (size_t)layout->h_words * sizeof(uint64_t));
    return scratch;
}

void tm_adapter_free_scratch(void *scratch_ptr) {
    TMAdapterScratch *scratch = (TMAdapterScratch *)scratch_ptr;
    if (scratch != NULL) {
        free(scratch->current);
        free(scratch);
    }
}

int32_t tm_adapter_predict_row(const TMLayout *layout, const float *row, void *scratch_ptr, int32_t *votes) {
    TMAdapterScratch *scratch = (TMAdapterScratch *)scratch_ptr;
    return tm_predict_row(layout, row, scratch->current, votes);
}

static void current_to_bytes_profile(const uint64_t *current, uint8_t *current_bytes, uint32_t n_bytes) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    memcpy(current_bytes, current, n_bytes);
#else
    for (uint32_t b = 0; b < n_bytes; b++) {
        current_bytes[b] = (uint8_t)(current[b >> 3] >> (8u * (b & 7u)));
    }
#endif
}

static double measure_binarize_backend(TMLayout *layout, const Matrix *x, uint32_t n_cal, uint32_t repeats, uint8_t prefer_feature_blocks, uint64_t *current, volatile uint64_t *sink) {
    layout->prefer_feature_blocks = prefer_feature_blocks;
    for (uint32_t i = 0; i < n_cal && i < TM_CALIBRATION_WARMUP_ROWS; i++) {
        tm_binarize_row(layout, x->x + (size_t)(i % x->n_rows) * x->n_features, current);
        *sink ^= current[0];
    }
    const double t0 = now_us();
    for (uint32_t r = 0; r < repeats; r++) {
        for (uint32_t i = 0; i < n_cal; i++) {
            tm_binarize_row(layout, x->x + (size_t)(i % x->n_rows) * x->n_features, current);
            *sink ^= current[0];
        }
    }
    return (now_us() - t0) / (double)(n_cal * repeats);
}

static void calibrate_binarize_backend(const Matrix *x, TMLayout *layout, int verbose) {
    if (layout->feature_blocks == NULL || layout->n_feature_blocks == 0u) {
        layout->prefer_feature_blocks = 0u;
        return;
    }
    const uint32_t n_cal = x->n_rows < TM_CALIBRATION_MAX_ROWS ? x->n_rows : TM_CALIBRATION_MAX_ROWS;
    uint64_t *current = xaligned_alloc(64, (size_t)layout->h_words * sizeof(uint64_t));
    volatile uint64_t sink = 0;
    double literal_us = 1.0e300;
    double feature_us = 1.0e300;
    for (uint32_t trial = 0; trial < TM_CALIBRATION_TRIALS; trial++) {
        double t = measure_binarize_backend(layout, x, n_cal, TM_CALIBRATION_REPEATS, (uint8_t)(trial & 1u), current, &sink);
        if ((trial & 1u) == 0u) {
            if (t < literal_us) literal_us = t;
        } else {
            if (t < feature_us) feature_us = t;
        }
        t = measure_binarize_backend(layout, x, n_cal, TM_CALIBRATION_REPEATS, (uint8_t)((trial & 1u) == 0u), current, &sink);
        if ((trial & 1u) == 0u) {
            if (t < feature_us) feature_us = t;
        } else {
            if (t < literal_us) literal_us = t;
        }
    }
    layout->prefer_feature_blocks = feature_us < literal_us * TM_BINARIZE_SELECT_MARGIN ? 1u : 0u;
    if (verbose) {
        printf("      calibrate: literal_binarize=%7.4f us  feature_binarize=%7.4f us"
               "  blocks=%u  selected=%s\n",
               literal_us, feature_us, layout->n_feature_blocks,
               layout->prefer_feature_blocks ? "feature" : "literal");
    }
    if (sink == UINT64_MAX) {
        printf("      binarize_sink=%" PRIu64 "\n", (uint64_t)sink);
    }
    free(current);
}

static double measure_score_backend(TMLayout *layout, const uint64_t *currents, uint32_t n_cal, uint32_t repeats, uint8_t prefer_delta, int32_t *votes, volatile int32_t *sink) {
    layout->prefer_delta_table = prefer_delta;
    for (uint32_t i = 0; i < n_cal && i < TM_CALIBRATION_WARMUP_ROWS; i++) {
        *sink ^= tm_predict_current(layout, currents + (size_t)i * layout->h_words, votes);
    }
    const double t0 = now_us();
    for (uint32_t r = 0; r < repeats; r++) {
        for (uint32_t i = 0; i < n_cal; i++) {
            *sink ^= tm_predict_current(layout, currents + (size_t)i * layout->h_words, votes);
        }
    }
    return (now_us() - t0) / (double)(n_cal * repeats);
}

static void calibrate_score_backend(const Matrix *x, TMLayout *layout, int verbose) {
    layout->prefer_delta_table = 0u;
    if (layout->byte_delta_tables == NULL || layout->base_mismatch_zero == NULL) {
        return;
    }
    const uint32_t n_cal = x->n_rows < TM_CALIBRATION_MAX_ROWS ? x->n_rows : TM_CALIBRATION_MAX_ROWS;
    uint64_t *currents = xaligned_alloc(64, (size_t)n_cal * layout->h_words * sizeof(uint64_t));
    for (uint32_t i = 0; i < n_cal; i++) {
        tm_binarize_row(layout, x->x + (size_t)(i % x->n_rows) * x->n_features,
                        currents + (size_t)i * layout->h_words);
    }

    uint64_t nonzero_bytes = 0;
    uint8_t current_bytes[TM_MAX_H_WORDS * 8u];
    for (uint32_t i = 0; i < n_cal; i++) {
        current_to_bytes_profile(currents + (size_t)i * layout->h_words, current_bytes, layout->n_bytes);
        for (uint32_t b = 0; b < layout->n_bytes; b++) {
            nonzero_bytes += current_bytes[b] != 0u ? 1u : 0u;
        }
    }
    const uint64_t total_bytes = (uint64_t)n_cal * layout->n_bytes;
#if (defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE) && !defined(TM_NO_AVX2_TABLE_SCORE)) || \
    (defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE) && !defined(TM_NO_NEON_TABLE_SCORE))
    const int vector_table_available = 1;
#else
    const int vector_table_available = 0;
#endif
    const int can_delta =
        vector_table_available &&
        layout->delta_table_exact != 0u &&
        nonzero_bytes * TM_DELTA_DENSE_BYTES_DENOMINATOR <= total_bytes * TM_DELTA_DENSE_BYTES_NUMERATOR;

    int32_t *votes = xaligned_alloc(64, (size_t)layout->n_classes * sizeof(int32_t));
    volatile int32_t sink = 0;
    double table_us = 1.0e300;
    double delta_us = 1.0e300;
    for (uint32_t trial = 0; trial < TM_CALIBRATION_TRIALS; trial++) {
        double t = measure_score_backend(layout, currents, n_cal, TM_CALIBRATION_REPEATS, 0u, votes, &sink);
        if (t < table_us) table_us = t;
        if (can_delta) {
            t = measure_score_backend(layout, currents, n_cal, TM_CALIBRATION_REPEATS, 1u, votes, &sink);
            if (t < delta_us) delta_us = t;
        }
    }
    const char *selected = "table";
    if (can_delta && delta_us < table_us * TM_DELTA_SELECT_MARGIN) {
        layout->prefer_delta_table = 1u;
        selected = "delta";
    }
    if (verbose) {
        if (can_delta) {
            printf("      calibrate: table_score=%7.4f us  delta_score=%7.4f us  selected=%s\n",
                   table_us, delta_us, selected);
        } else {
            printf("      calibrate: table_score=%7.4f us  selected=table  reason=dense_current\n", table_us);
        }
        if (sink == INT32_MIN) {
            printf("      score_sink=%" PRId32 "\n", (int32_t)sink);
        }
    }
    free(votes);
    free(currents);
}

void tm_adapter_calibrate_layout(const Matrix *x, TMLayout *layout, int verbose) {
    calibrate_binarize_backend(x, layout, verbose);
    calibrate_score_backend(x, layout, verbose);
}

static const char *selected_score_backend(const TMLayout *l, uint32_t *lane_width) {
#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE) && !defined(TM_NO_AVX2_TABLE_SCORE)
    if (l->byte_tables != NULL && l->table_stride >= l->n_clauses + 31u) {
        *lane_width = 32u;
        return l->prefer_delta_table ? "delta_table_avx2" : "table_avx2";
    }
#endif
#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE) && !defined(TM_NO_NEON_TABLE_SCORE)
    if (l->byte_tables != NULL && l->table_stride >= l->n_clauses + 15u) {
        *lane_width = 16u;
        return l->prefer_delta_table ? "delta_table_neon" : "table_neon";
    }
#endif
    *lane_width = 1u;
    return "unavailable";
}

void tm_adapter_print_profile(const TMLayout *layout, const Matrix *x, void *scratch_ptr, int32_t *votes, int detail) {
    TMAdapterScratch *scratch = (TMAdapterScratch *)scratch_ptr;
    const uint32_t n_time = x->n_rows < TM_PROFILE_MAX_ROWS ? x->n_rows : TM_PROFILE_MAX_ROWS;
    uint64_t *currents = xaligned_alloc(64, (size_t)n_time * layout->h_words * sizeof(uint64_t));

    double t0 = now_us();
    for (uint32_t i = 0; i < n_time; i++) {
        tm_binarize_row(layout, x->x + (size_t)(i % x->n_rows) * x->n_features,
                        currents + (size_t)i * layout->h_words);
    }
    const double bin_us = (now_us() - t0) / (double)n_time;

    t0 = now_us();
    for (uint32_t i = 0; i < n_time; i++) {
        (void)tm_predict_current(layout, currents + (size_t)i * layout->h_words, votes);
    }
    const double score_us = (now_us() - t0) / (double)n_time;

    t0 = now_us();
    for (uint32_t i = 0; i < n_time; i++) {
        (void)tm_predict_row(layout, x->x + (size_t)(i % x->n_rows) * x->n_features, scratch->current, votes);
    }
    const double total_us = (now_us() - t0) / (double)n_time;

    printf("      profile: binarize=%7.3f us  score=%7.3f us  total=%7.3f us\n",
           bin_us, score_us, total_us);

    if (detail) {
        uint32_t lane_width = 1u;
        const char *backend = selected_score_backend(layout, &lane_width);
        uint64_t blocks_per_row = 0;
        uint64_t scored_clauses_per_row = 0;
        for (uint16_t k = 0; k < layout->n_classes; k++) {
            const uint32_t pos_start = (uint32_t)layout->pos_start[k];
            const uint32_t pos_end = (uint32_t)layout->pos_end[k];
            const uint32_t neg_start = (uint32_t)layout->neg_start[k];
            const uint32_t neg_end = (uint32_t)layout->neg_end[k];
            blocks_per_row += ceil_div_u32(pos_end - pos_start, lane_width) +
                              ceil_div_u32(neg_end - neg_start, lane_width);
            scored_clauses_per_row += (uint64_t)(pos_end - pos_start) + (uint64_t)(neg_end - neg_start);
        }
        printf("      detail: backend=%s  binarize=%s  lane=%u  rows=%u"
               "  n_bytes=%u  table_stride=%u\n",
               backend,
               layout->prefer_feature_blocks ? "feature" : "literal",
               lane_width,
               n_time,
               layout->n_bytes,
               layout->table_stride);
        printf("              score_work: blocks/row=%" PRIu64
               "  lanes/row=%" PRIu64 "  clauses/row=%" PRIu64 "\n",
               blocks_per_row,
               blocks_per_row * lane_width,
               scored_clauses_per_row);
        for (uint16_t k = 0; k < layout->n_classes; k++) {
            const uint32_t pos_start = (uint32_t)layout->pos_start[k];
            const uint32_t pos_end = (uint32_t)layout->pos_end[k];
            const uint32_t neg_start = (uint32_t)layout->neg_start[k];
            const uint32_t neg_end = (uint32_t)layout->neg_end[k];
            printf("              class[%u]: pos=%u  neg=%u  blocks=%u\n",
                   (unsigned)k,
                   pos_end - pos_start,
                   neg_end - neg_start,
                   ceil_div_u32(pos_end - pos_start, lane_width) +
                   ceil_div_u32(neg_end - neg_start, lane_width));
        }
    }
    free(currents);
}

static TMModelStats collect_model_stats(const FBZModel *m, const TMLayout *l) {
    TMModelStats stats;
    memset(&stats, 0, sizeof(stats));
    uint64_t used[TM_MAX_H_WORDS];
    memset(used, 0, sizeof(used));
    uint64_t total_active_literals = 0;
    uint64_t total_active_words = 0;
    for (uint32_t c = 0; c < m->n_clauses; c++) {
        uint32_t active_literals = 0;
        uint32_t active_words = 0;
        for (uint32_t h = 0; h < m->h_words; h++) {
            uint64_t word = (m->lits[(size_t)c * m->h_words + h] |
                             m->inv[(size_t)c * m->h_words + h]) &
                            valid_literal_mask(h, m->n_literals);
            if (word != 0) {
                active_words++;
                active_literals += popcount64_u32(word);
                used[h] |= word;
            }
        }
        total_active_literals += active_literals;
        total_active_words += active_words;
        stats.total_postings += active_literals;
        stats.empty_clauses += active_literals == 0 ? 1u : 0u;
        stats.linearizable_clauses += (uint32_t)m->clamp[c] >= active_literals ? 1u : 0u;
        stats.clamp1_clauses += m->clamp[c] == 1 ? 1u : 0u;
    }
    uint32_t used_literals = 0;
    for (uint32_t h = 0; h < m->h_words; h++) {
        used_literals += popcount64_u32(used[h] & valid_literal_mask(h, m->n_literals));
    }
    stats.unused_literals = (uint32_t)m->n_literals - used_literals;
    if (m->n_clauses != 0) {
        stats.avg_active_literals_per_clause = (double)total_active_literals / (double)m->n_clauses;
        stats.avg_active_words_per_clause = (double)total_active_words / (double)m->n_clauses;
    }
    stats.dense_bytes = (uint64_t)m->n_clauses * 2u * m->h_words * sizeof(uint64_t);
    stats.byte_table_bytes = (uint64_t)l->n_bytes * 256u * l->table_stride * sizeof(uint8_t);
    stats.delta_table_bytes = (uint64_t)l->n_bytes * 256u * l->table_stride * sizeof(int8_t);
    stats.base_mismatch_bytes = (uint64_t)l->table_stride * sizeof(uint8_t);
    return stats;
}

void tm_adapter_print_stats(const FBZModel *m, const TMLayout *l) {
    TMModelStats stats = collect_model_stats(m, l);
    printf("      stats: active_lits_avg=%6.2f  active_words_avg=%5.2f"
           "  empty=%u  linearizable=%u  clamp1=%u  unused_literals=%u\n",
           stats.avg_active_literals_per_clause,
           stats.avg_active_words_per_clause,
           stats.empty_clauses,
           stats.linearizable_clauses,
           stats.clamp1_clauses,
           stats.unused_literals);
    printf("             total_postings=%" PRIu64 "  dense_bytes=%" PRIu64
           "  byte_table_bytes=%" PRIu64 "\n",
           stats.total_postings,
           stats.dense_bytes,
           stats.byte_table_bytes);
    printf("             delta_table_bytes=%" PRIu64 "  base_mismatch_bytes=%" PRIu64 "\n",
           stats.delta_table_bytes,
           stats.base_mismatch_bytes);
}

const char *tm_adapter_profile_label(void) {
    return "tm_v17 byte/delta table backend";
}

const char *tm_adapter_usage_suffix(void) {
    return "";
}
