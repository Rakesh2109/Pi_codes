#define _POSIX_C_SOURCE 200809L

#include "tm_common.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <zstd.h>

#define TM_PREDICT_WARMUP_ROWS 5000u
#define TM_PROFILE_MAX_ROWS 3000u

static const Dataset DATASETS[] = {
    {"wustl", "WUSTL", "wustl_test"},
    {"nslkdd", "NSLKDD", "nslkdd_test"},
    {"toniot", "TonIoT", "toniot_test"},
    {"medsec", "MedSec", "medsec_test"},
};

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", n);
        exit(1);
    }
    return p;
}

void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "out of memory allocating %zu x %zu bytes\n", count, size);
        exit(1);
    }
    return p;
}

void *xaligned_alloc(size_t alignment, size_t n) {
    void *p = NULL;
    int err = posix_memalign(&p, alignment, n ? n : alignment);
    if (err || !p) {
        fprintf(stderr, "aligned allocation failed for %zu bytes\n", n);
        exit(1);
    }
    return p;
}

double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

int32_t read_i32_le(const uint8_t *p) {
    return (int32_t)read_u32_le(p);
}

float read_f32_le(const uint8_t *p) {
    uint32_t bits = read_u32_le(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint64_t read_u64_from_le_bytes(const uint8_t *p, uint32_t available) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < available; i++) {
        value |= (uint64_t)p[i] << (8u * i);
    }
    return value;
}

uint32_t popcount64_u32(uint64_t v) {
    return (uint32_t)__builtin_popcountll(v);
}

uint64_t valid_literal_mask(uint32_t word_index, uint16_t n_literals) {
    uint32_t first = word_index << 6;
    if (first >= n_literals) {
        return 0;
    }
    uint32_t valid = (uint32_t)n_literals - first;
    if (valid >= 64u) {
        return UINT64_MAX;
    }
    return (UINT64_C(1) << valid) - UINT64_C(1);
}

uint32_t ceil_div_u32(uint32_t n, uint32_t d) {
    return d ? (n + d - 1u) / d : 0u;
}

uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek %s\n", path);
        exit(1);
    }
    long end = ftell(f);
    if (end < 0) {
        fprintf(stderr, "cannot size %s\n", path);
        exit(1);
    }
    rewind(f);
    uint8_t *buf = xmalloc((size_t)end);
    if (fread(buf, 1, (size_t)end, f) != (size_t)end) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    *out_size = (size_t)end;
    return buf;
}

static void skip_string_table(const uint8_t *blob, size_t blob_size, size_t *off) {
    if (*off + 2 > blob_size) {
        fprintf(stderr, "truncated FBZ string table\n");
        exit(1);
    }
    uint16_t n_strings = read_u16_le(blob + *off);
    *off += 2;
    for (uint16_t i = 0; i < n_strings; i++) {
        if (*off + 2 > blob_size) {
            fprintf(stderr, "truncated FBZ string entry\n");
            exit(1);
        }
        uint16_t len = read_u16_le(blob + *off);
        *off += 2;
        if (*off + len > blob_size) {
            fprintf(stderr, "truncated FBZ string payload\n");
            exit(1);
        }
        *off += len;
    }
}

FBZModel read_fbz(const char *path) {
    size_t blob_size = 0;
    uint8_t *blob = read_file(path, &blob_size);
    if (blob_size < 24 || memcmp(blob, "FBZ1", 4) != 0 || blob[4] != 1) {
        fprintf(stderr, "%s is not an FBZ1 model\n", path);
        exit(1);
    }

    FBZModel m;
    memset(&m, 0, sizeof(m));
    m.n_literals = read_u16_le(blob + 5);
    m.n_classes = read_u16_le(blob + 7);
    if (m.n_literals > TM_MAX_LITERALS) {
        fprintf(stderr,
                "%s has %u literals, but this C build supports at most %u literals\n",
                path, (unsigned)m.n_literals, (unsigned)TM_MAX_LITERALS);
        exit(1);
    }
    uint32_t total_hint = read_u32_le(blob + 10);
    uint32_t comp_size = read_u32_le(blob + 14);
    uint32_t uncomp_size = read_u32_le(blob + 18);
    m.h_words = (m.n_literals + 63u) / 64u;

    size_t off = 22;
    m.feat_idx = xmalloc((size_t)m.n_literals * sizeof(int32_t));
    m.thresh = xmalloc((size_t)m.n_literals * sizeof(float));
    for (uint16_t i = 0; i < m.n_literals; i++, off += 4) {
        m.feat_idx[i] = read_i32_le(blob + off);
    }
    for (uint16_t i = 0; i < m.n_literals; i++, off += 4) {
        m.thresh[i] = read_f32_le(blob + off);
    }
    skip_string_table(blob, blob_size, &off);
    skip_string_table(blob, blob_size, &off);
    if (off + comp_size > blob_size) {
        fprintf(stderr, "truncated FBZ zstd payload in %s\n", path);
        exit(1);
    }

    uint8_t *payload = xmalloc(uncomp_size);
    size_t got = ZSTD_decompress(payload, uncomp_size, blob + off, comp_size);
    if (ZSTD_isError(got) || got != uncomp_size) {
        fprintf(stderr, "zstd decompress failed for %s: %s\n",
                path, ZSTD_getErrorName(got));
        exit(1);
    }

    uint32_t capacity = total_hint ? total_hint : 16u;
    m.lits = xmalloc((size_t)capacity * m.h_words * sizeof(uint64_t));
    m.inv = xmalloc((size_t)capacity * m.h_words * sizeof(uint64_t));
    m.clamp = xmalloc((size_t)capacity * sizeof(int32_t));
    m.sign = xmalloc((size_t)capacity * sizeof(int32_t));
    m.cls = xmalloc((size_t)capacity * sizeof(int32_t));

    uint32_t chunk_bytes = (m.n_literals + 7u) / 8u;
    size_t boff = 0;
    for (uint16_t k = 0; k < m.n_classes; k++) {
        for (int polarity = 0; polarity < 2; polarity++) {
            int32_t sign = polarity == 0 ? 1 : -1;
            if (boff + 2 > uncomp_size) {
                fprintf(stderr, "truncated FBZ clause count\n");
                exit(1);
            }
            uint16_t n_clauses = read_u16_le(payload + boff);
            boff += 2;
            for (uint16_t c = 0; c < n_clauses; c++) {
                if (boff + 1 + 2u * chunk_bytes > uncomp_size) {
                    fprintf(stderr, "truncated FBZ clause payload\n");
                    exit(1);
                }
                if (m.n_clauses == capacity) {
                    capacity *= 2u;
                    m.lits = realloc(m.lits, (size_t)capacity * m.h_words * sizeof(uint64_t));
                    m.inv = realloc(m.inv, (size_t)capacity * m.h_words * sizeof(uint64_t));
                    m.clamp = realloc(m.clamp, (size_t)capacity * sizeof(int32_t));
                    m.sign = realloc(m.sign, (size_t)capacity * sizeof(int32_t));
                    m.cls = realloc(m.cls, (size_t)capacity * sizeof(int32_t));
                    if (!m.lits || !m.inv || !m.clamp || !m.sign || !m.cls) {
                        fprintf(stderr, "out of memory growing clauses\n");
                        exit(1);
                    }
                }

                uint32_t row = m.n_clauses++;
                m.clamp[row] = payload[boff++];
                const uint8_t *pos_raw = payload + boff;
                boff += chunk_bytes;
                const uint8_t *neg_raw = payload + boff;
                boff += chunk_bytes;
                for (uint32_t h = 0; h < m.h_words; h++) {
                    uint32_t remaining = chunk_bytes > h * 8u ? chunk_bytes - h * 8u : 0u;
                    uint32_t take = remaining < 8u ? remaining : 8u;
                    m.lits[(size_t)row * m.h_words + h] =
                        read_u64_from_le_bytes(pos_raw + h * 8u, take);
                    m.inv[(size_t)row * m.h_words + h] =
                        read_u64_from_le_bytes(neg_raw + h * 8u, take);
                }
                m.sign[row] = sign;
                m.cls[row] = k;
            }
        }
    }

    free(payload);
    free(blob);
    return m;
}

Matrix load_x(const char *path) {
    size_t bytes = 0;
    uint8_t *buf = read_file(path, &bytes);
    if (bytes < 8) {
        fprintf(stderr, "truncated X file %s\n", path);
        exit(1);
    }
    Matrix x;
    x.n_rows = read_u32_le(buf);
    x.n_features = read_u32_le(buf + 4);
    size_t expected = 8u + (size_t)x.n_rows * x.n_features * sizeof(float);
    if (bytes != expected) {
        fprintf(stderr, "bad X size for %s: got %zu expected %zu\n", path, bytes, expected);
        exit(1);
    }
    x.x = xmalloc((size_t)x.n_rows * x.n_features * sizeof(float));
    const uint8_t *p = buf + 8;
    for (uint64_t i = 0; i < (uint64_t)x.n_rows * x.n_features; i++) {
        x.x[i] = read_f32_le(p + i * 4u);
    }
    free(buf);
    return x;
}

int32_t *load_y(const char *path, uint32_t n_rows) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    int32_t *y = xmalloc((size_t)n_rows * sizeof(int32_t));
    for (uint32_t i = 0; i < n_rows; i++) {
        if (fscanf(f, "%" SCNd32, &y[i]) != 1) {
            fprintf(stderr, "bad label at row %u in %s\n", i, path);
            exit(1);
        }
    }
    fclose(f);
    return y;
}

double macro_f1(const int32_t *y_true, const int32_t *y_pred, uint32_t n, uint16_t k) {
    double total = 0.0;
    for (uint16_t cls = 0; cls < k; cls++) {
        uint32_t tp = 0, fp = 0, fn = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (y_pred[i] == cls && y_true[i] == cls) {
                tp++;
            } else if (y_pred[i] == cls && y_true[i] != cls) {
                fp++;
            } else if (y_pred[i] != cls && y_true[i] == cls) {
                fn++;
            }
        }
        double p = (tp + fp) ? (double)tp / (double)(tp + fp) : 0.0;
        double r = (tp + fn) ? (double)tp / (double)(tp + fn) : 0.0;
        total += (p + r) ? (2.0 * p * r / (p + r)) : 0.0;
    }
    return total / (double)k;
}

void free_fbz(FBZModel *m) {
    free(m->feat_idx);
    free(m->thresh);
    free(m->lits);
    free(m->inv);
    free(m->clamp);
    free(m->sign);
    free(m->cls);
}

void write_predictions(const char *pred_dir, const char *stem, const int32_t *y_pred, uint32_t n_rows) {
    if (!pred_dir) {
        return;
    }
    if (mkdir(pred_dir, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "cannot create prediction directory %s: %s\n",
                pred_dir, strerror(errno));
        exit(1);
    }
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_pred.txt", pred_dir, stem);
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno));
        exit(1);
    }
    for (uint32_t i = 0; i < n_rows; i++) {
        fprintf(f, "%" PRId32 "\n", y_pred[i]);
    }
    fclose(f);
}

#ifndef TM_COMMON_NO_MAIN
static int directory_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static const char *default_assets_dir(void) {
    if (directory_exists("assets")) {
        return "assets";
    }
    if (directory_exists("../assets")) {
        return "../assets";
    }
    if (directory_exists("../../assets")) {
        return "../../assets";
    }
    if (directory_exists("../../../assets")) {
        return "../../../assets";
    }
    if (directory_exists("../../../../assets")) {
        return "../../../../assets";
    }
    if (directory_exists("src/fuzzy_tm_infer/assets")) {
        return "src/fuzzy_tm_infer/assets";
    }
    return "assets";
}

static void run_dataset(
    const char *assets_dir,
    const char *pred_dir,
    int profile,
    int profile_detail,
    int stats,
    const Dataset *ds
) {
    char model_path[512];
    char x_path[512];
    char y_path[512];
    snprintf(model_path, sizeof(model_path), "%s/tm_models/%s_model.fbz", assets_dir, ds->stem);
    snprintf(x_path, sizeof(x_path), "%s/datasets/%s/%s_X_test_raw.bin", assets_dir, ds->test_dir, ds->name);
    snprintf(y_path, sizeof(y_path), "%s/datasets/%s/%s_Y_test.txt", assets_dir, ds->test_dir, ds->name);

    FBZModel fbz = read_fbz(model_path);
    Matrix x = load_x(x_path);
    TMLayout layout = tm_adapter_build_layout(&fbz);
    tm_adapter_calibrate_layout(&x, &layout, profile_detail);

    int32_t *y = load_y(y_path, x.n_rows);
    int32_t *y_pred = xmalloc((size_t)x.n_rows * sizeof(int32_t));
    int32_t *votes = xaligned_alloc(64, (size_t)layout.n_classes * sizeof(int32_t));
    void *scratch = tm_adapter_create_scratch(&layout);

    for (uint32_t i = 0; i < TM_PREDICT_WARMUP_ROWS && i < x.n_rows; i++) {
        (void)tm_adapter_predict_row(&layout, x.x + (size_t)(i % x.n_rows) * x.n_features, scratch, votes);
    }

    for (uint32_t i = 0; i < x.n_rows; i++) {
        y_pred[i] = tm_adapter_predict_row(&layout, x.x + (size_t)i * x.n_features, scratch, votes);
    }
    write_predictions(pred_dir, ds->stem, y_pred, x.n_rows);

    uint32_t correct = 0;
    for (uint32_t i = 0; i < x.n_rows; i++) {
        correct += y_pred[i] == y[i] ? 1u : 0u;
    }
    double acc = (double)correct / (double)x.n_rows;
    double f1 = macro_f1(y, y_pred, x.n_rows, layout.n_classes);

    uint32_t n_time = x.n_rows < TM_PROFILE_MAX_ROWS ? x.n_rows : TM_PROFILE_MAX_ROWS;
    double t0 = now_us();
    for (uint32_t i = 0; i < n_time; i++) {
        (void)tm_adapter_predict_row(&layout, x.x + (size_t)(i % x.n_rows) * x.n_features, scratch, votes);
    }
    double us = (now_us() - t0) / (double)n_time;

    printf("  %-10s  %2u  %4u  %3u  %9.3f   %6.4f   %6.4f\n",
           ds->name, layout.h_words, layout.n_literals, layout.n_classes, us, acc, f1);
    if (profile || profile_detail) {
        tm_adapter_print_profile(&layout, &x, scratch, votes, profile_detail);
    }
    if (stats) {
        tm_adapter_print_stats(&fbz, &layout);
    }

    tm_adapter_free_scratch(scratch);
    free(votes);
    free(y_pred);
    free(y);
    free(x.x);
    tm_adapter_free_layout(&layout);
    free_fbz(&fbz);
}

static void usage(const char *argv0) {
    printf("Usage: %s [assets_dir] [prediction_dir] [--profile] [--profile-detail] [--stats]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --profile          Print timing details.\n");
    printf("  --profile-detail   Print calibration/backend work-shape details.\n");
    printf("  --stats            Print model/layout statistics.\n");
    printf("  --cache/--cache-auto/--group-cache\n");
    printf("                     Accepted for compatibility; the shared runner scores rows directly.\n");
    printf("%s", tm_adapter_usage_suffix());
    printf("  -h, --help         Show this help.\n");
}

int main(int argc, char **argv) {
    const char *assets_dir = default_assets_dir();
    const char *pred_dir = NULL;
    int profile = 0;
    int profile_detail = 0;
    int stats = 0;
    int arg_start = 1;
    if (argc > 1 && strncmp(argv[1], "--", 2) != 0) {
        assets_dir = argv[1];
        arg_start = 2;
    }
    for (int i = arg_start; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) {
            profile = 1;
        } else if (strcmp(argv[i], "--profile-detail") == 0) {
            profile_detail = 1;
        } else if (strcmp(argv[i], "--stats") == 0) {
            stats = 1;
        } else if (
            strcmp(argv[i], "--cache") == 0 ||
            strcmp(argv[i], "--cache-auto") == 0 ||
            strcmp(argv[i], "--group-cache") == 0
        ) {
            continue;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            pred_dir = argv[i];
        }
    }

    printf("==============================================================================\n");
    printf("  PURE-C TM INFERENCE BENCHMARK (%s)\n", tm_adapter_profile_label());
    printf("==============================================================================\n\n");
    printf("  %-10s  %2s  %4s  %3s  %10s  %7s  %7s\n",
           "Dataset", "H", "N", "K", "us/sample", "Acc", "F1");
    printf("  -------------------------------------------------------\n");

    for (size_t i = 0; i < sizeof(DATASETS) / sizeof(DATASETS[0]); i++) {
        run_dataset(assets_dir, pred_dir, profile, profile_detail, stats, &DATASETS[i]);
    }
    return 0;
}
#endif
