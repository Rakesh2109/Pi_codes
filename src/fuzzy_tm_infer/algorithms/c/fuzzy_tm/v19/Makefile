CC ?= gcc
CPPFLAGS ?= -I. -I../../common
BASE_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -funroll-loops
ARCHFLAGS ?= -march=native
V19_FLAGS ?= -DTM_ENABLE_SCORE_BLOCKS -DTM_ENABLE_FEATURE_STATE_TABLES -DTM_SCORE_BLOCKS_METADATA_ONLY
CFLAGS ?= $(BASE_CFLAGS) $(ARCHFLAGS) $(V19_FLAGS)
LDLIBS ?= -lzstd

.PHONY: all clean run profile profile-detail lib

all: tm_infer_c

tm_infer_c: tm_infer_c.c tm_adapter.c tm_algorithm.c tm_algorithm.h ../../common/tm_common.h ../../common/tm_infer_common.c
	$(CC) $(CPPFLAGS) $(CFLAGS) tm_infer_c.c tm_adapter.c tm_algorithm.c -o $@ $(LDLIBS)

lib: libfuzzy_tm_infer.so

libfuzzy_tm_infer.so: tm_adapter.c tm_algorithm.c tm_algorithm.h ../../common/tm_common.h ../../common/tm_io_common.c ../../common/tm_c_api.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared ../../common/tm_io_common.c ../../common/tm_c_api.c tm_adapter.c tm_algorithm.c -o $@ $(LDLIBS)

run: tm_infer_c
	./tm_infer_c assets

profile: tm_infer_c
	./tm_infer_c assets --profile

profile-detail: tm_infer_c
	./tm_infer_c assets --profile-detail --stats

clean:
	rm -f tm_infer_c libfuzzy_tm_infer.so
