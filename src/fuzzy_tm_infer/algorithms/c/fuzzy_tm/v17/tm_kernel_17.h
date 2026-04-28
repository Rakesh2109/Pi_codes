#ifndef TM_KERNEL_17_H
#define TM_KERNEL_17_H

#include "tm_algorithm.h"

#if defined(__AVX2__) && !defined(TM_NO_AVX2_SCORE)
int32_t tm17_predict_current_avx2(
    const TMLayout *restrict layout,
    const uint64_t *restrict current,
    int32_t *restrict votes
);
#endif

#if defined(__ARM_NEON) && !defined(TM_NO_NEON_SCORE)
int32_t tm17_predict_current_neon(
    const TMLayout *restrict layout,
    const uint64_t *restrict current,
    int32_t *restrict votes
);
#endif

#endif
