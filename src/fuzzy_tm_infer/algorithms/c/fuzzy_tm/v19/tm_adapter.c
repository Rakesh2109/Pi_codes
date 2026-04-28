#include "../../common/tm_common.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TM_SCORE_BLOCK_LANES
#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
#define TM_SCORE_BLOCK_LANES 32u
#elif defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
#define TM_SCORE_BLOCK_LANES 16u
#else
#define TM_SCORE_BLOCK_LANES 1u
#endif
#endif

#define TM_PROFILE_MAX_ROWS 3000u

typedef struct {
    const FBZModel *model;
    uint32_t idx;
} ClauseOrder;

typedef struct {
    double avg_active_literals_per_clause;
    double avg_active_words_per_clause;
    uint32_t empty_clauses;
    uint32_t linearizable_clauses;
    uint32_t clamp1_clauses;
    uint32_t unused_literals;
    uint64_t total_postings;
    uint64_t dense_bytes;
    uint64_t score_block_bytes;
    uint64_t feature_state_table_bytes;
    uint32_t score_blocks;
    uint32_t feature_state_values;
} TMModelStats;

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

static uint32_t count_score_blocks_for_range(uint32_t n, uint32_t lanes) {
    return n == 0u ? 0u : ceil_div_u32(n, lanes);
}

static void add_score_block(TMLayout *l, uint16_t class_id, uint32_t start, uint32_t pos_end, uint32_t end) {
    TMScoreBlock *block = l->score_blocks + l->n_score_blocks;
    const uint32_t lanes = l->score_block_lanes;
    const uint32_t count = end - start < lanes ? end - start : lanes;
    memset(block, 0, sizeof(*block));
    block->start = start;
    block->class_id = class_id;
    block->lanes = (uint8_t)count;
    block->pos_lanes = (uint8_t)(pos_end > start
        ? ((pos_end - start) < count ? (pos_end - start) : count)
        : 0u);
    for (uint32_t lane = 0; lane < count; lane++) {
        const uint32_t c = start + lane;
        block->clamp[lane] = l->clamp[c];
        block->base[lane] = l->base_mismatch_zero[c];
    }
    l->n_score_blocks++;
}

static void build_score_blocks(TMLayout *l) {
    const uint32_t lanes = TM_SCORE_BLOCK_LANES;
    uint32_t max_blocks = 0;
    l->score_block_lanes = lanes;
    for (uint16_t k = 0; k < l->n_classes; k++) {
        const uint32_t pos_start = (uint32_t)l->pos_start[k];
        const uint32_t pos_end = (uint32_t)l->pos_end[k];
        const uint32_t neg_start = (uint32_t)l->neg_start[k];
        const uint32_t neg_end = (uint32_t)l->neg_end[k];
        const uint32_t start = pos_end > pos_start ? pos_start : neg_start;
        const uint32_t end = neg_end > neg_start ? neg_end : pos_end;
        max_blocks += count_score_blocks_for_range(end - start, lanes);
    }
    l->score_blocks = xaligned_alloc(64, (size_t)max_blocks * sizeof(TMScoreBlock));
    memset(l->score_blocks, 0, (size_t)max_blocks * sizeof(TMScoreBlock));
    for (uint16_t k = 0; k < l->n_classes; k++) {
        const uint32_t pos_start = (uint32_t)l->pos_start[k];
        const uint32_t pos_end = (uint32_t)l->pos_end[k];
        const uint32_t neg_start = (uint32_t)l->neg_start[k];
        const uint32_t neg_end = (uint32_t)l->neg_end[k];
        const uint32_t start = pos_end > pos_start ? pos_start : neg_start;
        const uint32_t end = neg_end > neg_start ? neg_end : pos_end;
        for (uint32_t c = start; c < end; c += lanes) {
            add_score_block(l, k, c, pos_end, end < c + lanes ? end : c + lanes);
        }
    }
}

static void build_feature_state_tables(TMLayout *l) {
    l->feature_state_offsets = xcalloc(l->n_feature_blocks + 1u, sizeof(uint32_t));
    for (uint32_t fb = 0; fb < l->n_feature_blocks; fb++) {
        l->feature_state_offsets[fb + 1u] =
            l->feature_state_offsets[fb] + (uint32_t)l->feature_blocks[fb].count + 1u;
    }
    l->n_feature_state_values = l->feature_state_offsets[l->n_feature_blocks];
    const uint32_t lanes = l->score_block_lanes;
    const size_t table_size = (size_t)l->n_score_blocks * l->n_feature_state_values * lanes;
    l->blocked_feature_state_delta_tables = xaligned_alloc(64, table_size * sizeof(int8_t));
    memset(l->blocked_feature_state_delta_tables, 0, table_size * sizeof(int8_t));

    for (uint32_t bi = 0; bi < l->n_score_blocks; bi++) {
        const TMScoreBlock *block = l->score_blocks + bi;
        for (uint32_t fb = 0; fb < l->n_feature_blocks; fb++) {
            const TMFeatureBlock *feature = l->feature_blocks + fb;
            const uint32_t offset = l->feature_state_offsets[fb];
            for (uint32_t state = 0; state <= feature->count; state++) {
                int8_t *row =
                    l->blocked_feature_state_delta_tables +
                    ((size_t)bi * l->n_feature_state_values + offset + state) * lanes;
                for (uint32_t lane = 0; lane < block->lanes; lane++) {
                    const uint32_t c = block->start + lane;
                    int32_t delta = 0;
                    for (uint32_t j = 0; j < state; j++) {
                        const uint32_t lit = (uint32_t)feature->literal_begin + j;
                        const uint32_t h = lit >> 6;
                        const uint32_t bit = lit & 63u;
                        const uint64_t target = l->inter[(size_t)c * 2u * l->h_words + 2u * h];
                        const uint64_t mask = l->inter[(size_t)c * 2u * l->h_words + 2u * h + 1u];
                        if (((mask >> bit) & UINT64_C(1)) != 0u) {
                            delta += ((target >> bit) & UINT64_C(1)) != 0u ? -1 : 1;
                        }
                    }
                    row[lane] = (int8_t)delta;
                }
            }
        }
    }
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
    for (uint32_t h = 0; h < l->h_words; h++) {
        const uint64_t lits =
            l->inter[(size_t)out * 2u * l->h_words + 2u * h] &
            valid_literal_mask(h, l->n_literals);
        base_mismatches_zero += popcount64_u32(lits);
    }
    l->base_mismatch_zero[out] = (uint8_t)base_mismatches_zero;
}

TMLayout tm_adapter_build_layout(const FBZModel *m) {
    TMLayout l;
    memset(&l, 0, sizeof(l));
    l.n_literals = m->n_literals;
    l.n_classes = m->n_classes;
    l.h_words = m->h_words;
    l.n_clauses = m->n_clauses;
    l.thresh = xmalloc((size_t)l.n_literals * sizeof(float));
    memcpy(l.thresh, m->thresh, (size_t)l.n_literals * sizeof(float));
    l.feature_blocks = build_feature_blocks(m, &l.n_feature_blocks);
    l.inter = xaligned_alloc(64, (size_t)l.n_clauses * 2u * l.h_words * sizeof(uint64_t));
    l.base_mismatch_zero = xcalloc(l.n_clauses, sizeof(uint8_t));
    l.clamp = xcalloc(l.n_clauses, sizeof(uint8_t));
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
    for (uint32_t out = 0; out < m->n_clauses; out++) {
        write_layout_clause(&l, m, out, order[out].idx, sorted_cls, sorted_sign);
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
    build_score_blocks(&l);
    build_feature_state_tables(&l);
    free(l.inter);
    l.inter = NULL;
    free(l.base_mismatch_zero);
    l.base_mismatch_zero = NULL;
    free(sorted_cls);
    free(sorted_sign);
    free(order);
    return l;
}

void tm_adapter_calibrate_layout(const Matrix *x, TMLayout *layout, int verbose) {
    (void)x;
    (void)layout;
    (void)verbose;
}

void tm_adapter_free_layout(TMLayout *l) {
    free(l->thresh);
    free(l->feature_blocks);
    free(l->clamp);
    free(l->pos_start);
    free(l->pos_end);
    free(l->neg_start);
    free(l->neg_end);
    free(l->score_blocks);
    free(l->feature_state_offsets);
    free(l->blocked_feature_state_delta_tables);
}

void *tm_adapter_create_scratch(const TMLayout *layout) {
    (void)layout;
    return NULL;
}

void tm_adapter_free_scratch(void *scratch) {
    (void)scratch;
}

int32_t tm_adapter_predict_row(const TMLayout *layout, const float *row, void *scratch, int32_t *votes) {
    (void)scratch;
    return tm_predict_row(layout, row, votes);
}

static const char *selected_score_backend(uint32_t *lane_width) {
#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
    *lane_width = 32u;
    return "feature_state_avx2";
#elif defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
    *lane_width = 16u;
    return "feature_state_neon";
#else
    *lane_width = 1u;
    return "unavailable";
#endif
}

void tm_adapter_print_profile(const TMLayout *layout, const Matrix *x, void *scratch, int32_t *votes, int detail) {
    (void)scratch;
    const uint32_t n_time = x->n_rows < TM_PROFILE_MAX_ROWS ? x->n_rows : TM_PROFILE_MAX_ROWS;
    const double t0 = now_us();
    for (uint32_t i = 0; i < n_time; i++) {
        (void)tm_predict_row(layout, x->x + (size_t)(i % x->n_rows) * x->n_features, votes);
    }
    const double total_us = (now_us() - t0) / (double)n_time;
    printf("      profile: row_infer=%7.3f us\n", total_us);
    if (!detail) {
        return;
    }

    uint32_t lane_width = 1u;
    const char *backend = selected_score_backend(&lane_width);
    const uint64_t table_rows_per_row = (uint64_t)layout->n_score_blocks * layout->n_feature_blocks;
    printf("      detail: backend=%s  lane=%u  rows=%u"
           "  feature_blocks=%u  score_blocks=%u\n",
           backend, lane_width, n_time, layout->n_feature_blocks, layout->n_score_blocks);
    printf("              work: feature_state_values=%u  table_rows/row=%" PRIu64
           "  table_bytes/row=%" PRIu64 "\n",
           layout->n_feature_state_values,
           table_rows_per_row,
           table_rows_per_row * lane_width);
    for (uint16_t k = 0; k < layout->n_classes; k++) {
        const uint32_t pos_start = (uint32_t)layout->pos_start[k];
        const uint32_t pos_end = (uint32_t)layout->pos_end[k];
        const uint32_t neg_start = (uint32_t)layout->neg_start[k];
        const uint32_t neg_end = (uint32_t)layout->neg_end[k];
        const uint32_t start = pos_end > pos_start ? pos_start : neg_start;
        const uint32_t end = neg_end > neg_start ? neg_end : pos_end;
        printf("              class[%u]: pos=%u  neg=%u  blocks=%u\n",
               (unsigned)k,
               pos_end - pos_start,
               neg_end - neg_start,
               ceil_div_u32(end - start, lane_width));
    }
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
    stats.score_blocks = l->n_score_blocks;
    stats.feature_state_values = l->n_feature_state_values;
    stats.score_block_bytes = (uint64_t)l->n_score_blocks * sizeof(TMScoreBlock);
    stats.feature_state_table_bytes =
        (uint64_t)l->n_score_blocks * l->n_feature_state_values * l->score_block_lanes * sizeof(int8_t);
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
           "  score_block_bytes=%" PRIu64 "\n",
           stats.total_postings,
           stats.dense_bytes,
           stats.score_block_bytes);
    printf("             score_blocks=%u  feature_state_values=%u"
           "  feature_state_table_bytes=%" PRIu64 "\n",
           stats.score_blocks,
           stats.feature_state_values,
           stats.feature_state_table_bytes);
}

const char *tm_adapter_profile_label(void) {
    return "tm_v19 feature-state backend";
}

const char *tm_adapter_usage_suffix(void) {
    return "";
}
