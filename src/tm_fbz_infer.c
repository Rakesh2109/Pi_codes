/*
 * TM FBZ inference — C, per-sample, single-threaded.
 * Reads flat model binary (from tm_dump_c_model.py) + X_test_raw.bin.
 *
 * Compile:  gcc -O3 -march=native -o tm_fbz_infer tm_fbz_infer.c -lm
 * Run:      ./tm_fbz_infer wustl WUSTL
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── popcount: uses hardware CNT on aarch64 with -march=native ─────────────*/
static inline int pc64(uint64_t x) { return __builtin_popcountll(x); }

/* ── timing ─────────────────────────────────────────────────────────────── */
static inline double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* ── model ──────────────────────────────────────────────────────────────── */
typedef struct {
    uint32_t  N, K, total, H;
    int32_t  *feat_idx;    /* [N] */
    float    *thresh;      /* [N] */
    uint64_t *lits;        /* [total * H] row-major */
    uint64_t *xor_pre;     /* [total * H] */
    int32_t  *clamp;       /* [total] */
    int32_t  *sign;        /* [total] */
    int32_t  *cls;         /* [total] */
} Model;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static Model load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

    Model m;
    uint32_t hdr[4];
    fread(hdr, sizeof(uint32_t), 4, f);
    m.N = hdr[0]; m.K = hdr[1]; m.total = hdr[2]; m.H = hdr[3];

    m.feat_idx = xmalloc(m.N * sizeof(int32_t));
    m.thresh   = xmalloc(m.N * sizeof(float));
    m.lits     = xmalloc(m.total * m.H * sizeof(uint64_t));
    m.xor_pre  = xmalloc(m.total * m.H * sizeof(uint64_t));
    m.clamp    = xmalloc(m.total * sizeof(int32_t));
    m.sign     = xmalloc(m.total * sizeof(int32_t));
    m.cls      = xmalloc(m.total * sizeof(int32_t));

    fread(m.feat_idx, sizeof(int32_t),  m.N,            f);
    fread(m.thresh,   sizeof(float),    m.N,            f);
    fread(m.lits,     sizeof(uint64_t), m.total * m.H,  f);
    fread(m.xor_pre,  sizeof(uint64_t), m.total * m.H,  f);
    fread(m.clamp,    sizeof(int32_t),  m.total,        f);
    fread(m.sign,     sizeof(int32_t),  m.total,        f);
    fread(m.cls,      sizeof(int32_t),  m.total,        f);
    fclose(f);
    return m;
}

/* ── binarize one float32 row → uint64 chunks ───────────────────────────── */
static void binarize(const float *row, const Model *m, uint64_t *chunks) {
    memset(chunks, 0, m->H * sizeof(uint64_t));
    for (uint32_t i = 0; i < m->N; i++)
        if (row[m->feat_idx[i]] >= m->thresh[i])
            chunks[i >> 6] |= (uint64_t)1 << (i & 63);
}

/* ── predict one sample ─────────────────────────────────────────────────── */
static int predict(const uint64_t *x, const Model *m) {
    int32_t votes[16] = {0};    /* K ≤ 16 */
    for (uint32_t c = 0; c < m->total; c++) {
        const uint64_t *lc = m->lits    + c * m->H;
        const uint64_t *xp = m->xor_pre + c * m->H;
        int mism = 0;
        for (uint32_t ch = 0; ch < m->H; ch++)
            mism += pc64(lc[ch] ^ (xp[ch] & x[ch]));
        int out = m->clamp[c] - mism;
        if (out > 0) votes[m->cls[c]] += m->sign[c] * out;
    }
    int best = 0;
    for (uint32_t k = 1; k < m->K; k++)
        if (votes[k] > votes[best]) best = k;
    return best;
}

/* ── load test data ─────────────────────────────────────────────────────── */
static float *load_X(const char *path, uint32_t *n, uint32_t *d) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fread(n, sizeof(uint32_t), 1, f);
    fread(d, sizeof(uint32_t), 1, f);
    float *X = xmalloc(*n * *d * sizeof(float));
    fread(X, sizeof(float), *n * *d, f);
    fclose(f);
    return X;
}

static int32_t *load_y(const char *path, uint32_t n) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int32_t *y = xmalloc(n * sizeof(int32_t));
    for (uint32_t i = 0; i < n; i++) fscanf(f, "%d", &y[i]);
    fclose(f);
    return y;
}

/* ── macro F1 ───────────────────────────────────────────────────────────── */
static void macro_f1(const int32_t *y_pred, const int32_t *y_true,
                     uint32_t n, uint32_t K,
                     double *out_f1, double *out_acc) {
    int *tp = calloc(K, sizeof(int));
    int *fp = calloc(K, sizeof(int));
    int *fn = calloc(K, sizeof(int));
    int correct = 0;
    for (uint32_t i = 0; i < n; i++) {
        int p = y_pred[i], t = y_true[i];
        if (p == t) { tp[p]++; correct++; }
        else        { fp[p]++; fn[t]++; }
    }
    double mf1 = 0.0;
    for (uint32_t k = 0; k < K; k++) {
        double prec = (tp[k]+fp[k]) ? (double)tp[k]/(tp[k]+fp[k]) : 0.0;
        double rec  = (tp[k]+fn[k]) ? (double)tp[k]/(tp[k]+fn[k]) : 0.0;
        mf1 += (prec+rec > 0) ? 2*prec*rec/(prec+rec) : 0.0;
    }
    *out_f1  = mf1 / K;
    *out_acc = (double)correct / n;
    free(tp); free(fp); free(fn);
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <stem> <PREFIX>\n", argv[0]);
        return 1;
    }
    const char *stem   = argv[1];
    const char *prefix = argv[2];

    char mpath[256], xpath[256], ypath[256];
    snprintf(mpath, sizeof(mpath), "/tmp/%s_c_model.bin",     stem);
    snprintf(xpath, sizeof(xpath), "/tmp/%s_X_test_raw.bin",  prefix);
    snprintf(ypath, sizeof(ypath), "/tmp/%s_Y_test.txt",      prefix);

    Model    m  = load_model(mpath);
    uint32_t n, d;
    float   *X  = load_X(xpath, &n, &d);
    int32_t *y  = load_y(ypath, n);

    uint64_t *chunks  = xmalloc(m.H * sizeof(uint64_t));
    int32_t  *y_pred  = xmalloc(n   * sizeof(int32_t));

    /* warmup */
    for (int i = 0; i < 500; i++) {
        binarize(X + (i % n) * d, &m, chunks);
        predict(chunks, &m);
    }

    /* timing */
    int N_TIME = n < 3000 ? n : 3000;
    double t0 = now_us();
    for (int i = 0; i < N_TIME; i++) {
        binarize(X + (i % n) * d, &m, chunks);
        y_pred[i % n] = predict(chunks, &m);
    }
    double total_us = now_us() - t0;
    double per_us   = total_us / N_TIME;

    /* correctness on full test set */
    for (uint32_t i = 0; i < n; i++) {
        binarize(X + i * d, &m, chunks);
        y_pred[i] = predict(chunks, &m);
    }
    double f1, acc;
    macro_f1(y_pred, y, n, m.K, &f1, &acc);

    printf("%-10s  C  %8.2f us  F1=%.4f  acc=%.4f  N=%u K=%u total_clauses=%u\n",
           stem, per_us, f1, acc, m.N, m.K, m.total);

    free(chunks); free(y_pred); free(X); free(y);
    return 0;
}
