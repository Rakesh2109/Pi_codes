/*
 * Shared benchmark/CLI runner for tm_v17.
 *
 * Algorithm-specific layout building, calibration, profiling, and cleanup live
 * in tm_adapter.c. The common runner owns only FBZ/data loading, metrics,
 * argument parsing, prediction output, and benchmark orchestration.
 */
#include "../../common/tm_infer_common.c"
