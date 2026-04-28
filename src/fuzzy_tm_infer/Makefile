.PHONY: benchmark verify-native verify-fast-glade profile-fast-glade clean-native clean-fast-glade scaffold-native help

NATIVE_FUZZY_TM_DIR := algorithms/c/fuzzy_tm
NATIVE_GLADE_DIR := algorithms/c/booleanizers/glade_v2
NATIVE_VERSIONS ?= v17 v19
BASE ?= v17

PYTHON ?= uv run --with numpy --with numba --with zstandard python

benchmark:
	$(PYTHON) tm_infer.py

native-%:
	$(MAKE) -C $(NATIVE_FUZZY_TM_DIR)/$* all lib

verify-native: $(addprefix native-,$(NATIVE_VERSIONS))
	$(PYTHON) scripts/verify_native.py

native-glade-v2:
	$(MAKE) -C $(NATIVE_GLADE_DIR) lib

profile-fast-glade:
	$(MAKE) -C $(NATIVE_GLADE_DIR) profile-lib

verify-fast-glade: native-glade-v2
	$(PYTHON) scripts/verify_fast_glade.py

clean-native:
	@for version in $(NATIVE_VERSIONS); do \
		$(MAKE) -C $(NATIVE_FUZZY_TM_DIR)/$$version clean; \
	done

clean-fast-glade:
	$(MAKE) -C $(NATIVE_GLADE_DIR) clean

scaffold-native:
	@test -n "$(VERSION)" || (echo "usage: make scaffold-native VERSION=v20 [BASE=v17]" >&2; exit 2)
	$(PYTHON) scripts/scaffold_native_backend.py --version "$(VERSION)" --base "$(BASE)"

help:
	@echo "Root fuzzy_tm_infer is Python-first."
	@echo "Targets:"
	@echo "  benchmark      Run Python/Numba inference benchmark"
	@echo "  native-vNN     Build self-contained Fuzzy TM vNN binary/library"
	@echo "  verify-native  Verify native wrappers against Python inference"
	@echo "  verify-fast-glade  Verify fast GLADE v2 transform"
	@echo "  profile-fast-glade Build fast GLADE with GLADE_V2_PROFILE"
	@echo "  clean-native   Remove native build outputs for NATIVE_VERSIONS"
	@echo "  clean-fast-glade Remove fast GLADE build outputs"
	@echo "  scaffold-native VERSION=v20 [BASE=v17]"
