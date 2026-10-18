CC ?= cc

CPPFLAGS := -Iinclude -Isrc/core -Isrc/io -Isrc/model
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wpointer-arith -ffp-contract=off

LDLIBS += -lm

BIN_DIR ?= bin
BIN := $(BIN_DIR)/inkling
TESTS := $(addprefix $(BIN_DIR)/,test_safetensors test_config test_json test_catalogue test_alloc test_bf16 test_nvfp4 test_math test_model_bind test_model_runtime model_probe model_cli_probe test_generate generation_probe)
HEADERS := $(wildcard include/inkling/*.h) src/io/inkling_json.h $(wildcard src/core/*.h src/model/*.h)
IO_SOURCES := $(wildcard src/io/*.c)
CORE_SOURCES := $(wildcard src/core/*.c)
MODEL_SOURCES := $(wildcard src/model/*.c)

CHECKPOINT_FIXTURES := tests/fixtures/checkpoint
CONFIG := $(CHECKPOINT_FIXTURES)/config.json
INDEX := $(CHECKPOINT_FIXTURES)/model.safetensors.index.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := $(CHECKPOINT_FIXTURES)/headers/model-00005-of-00009.safetensors

.PHONY: all test test-sanitize oracle-fixtures model-fixtures generation-fixtures test-model-real test-mutations clean
all: $(BIN)

$(BIN): src/cli/inkling_run.c $(IO_SOURCES) $(CORE_SOURCES) $(MODEL_SOURCES)
$(BIN_DIR)/test_safetensors: tests/test_safetensors.c src/io/inkling_safetensors.c src/io/inkling_json.c
$(BIN_DIR)/test_config: tests/test_config.c src/io/inkling_config.c src/io/inkling_json.c
$(BIN_DIR)/test_json: tests/test_json.c src/io/inkling_json.c
$(BIN_DIR)/test_catalogue: tests/test_catalogue.c src/io/inkling_index.c src/io/inkling_safetensors.c src/io/inkling_json.c
$(BIN_DIR)/test_alloc: tests/test_alloc.c $(BIN_DIR)/inkling_alloc_test.o
$(BIN_DIR)/test_bf16: tests/test_bf16.c src/core/inkling_bf16.c
$(BIN_DIR)/test_nvfp4: tests/test_nvfp4.c src/core/inkling_nvfp4.c src/core/inkling_alloc.c src/core/inkling_bf16.c src/io/inkling_json.c

$(BIN_DIR)/test_math: tests/test_math.c $(CORE_SOURCES) src/io/inkling_json.c
$(BIN_DIR)/test_model_bind: tests/test_model_bind.c src/model/inkling_bind.c $(IO_SOURCES) $(CORE_SOURCES)
$(BIN_DIR)/test_model_runtime: tests/test_model_runtime.c $(MODEL_SOURCES) $(IO_SOURCES) $(CORE_SOURCES)
$(BIN_DIR)/model_probe: tests/model_probe.c $(MODEL_SOURCES) $(IO_SOURCES) $(CORE_SOURCES)
$(BIN_DIR)/generation_probe: tests/generation_probe.c $(MODEL_SOURCES) $(IO_SOURCES) $(CORE_SOURCES)
$(BIN_DIR)/test_generate: tests/test_generate.c $(MODEL_SOURCES) $(IO_SOURCES) $(filter-out src/core/inkling_alloc.c,$(CORE_SOURCES)) $(BIN_DIR)/inkling_alloc_test.o
$(BIN_DIR)/model_cli_probe: $(BIN_DIR)/model_cli_probe.o $(MODEL_SOURCES) $(IO_SOURCES) $(CORE_SOURCES)

$(BIN) $(TESTS): $(HEADERS) | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(filter %.c %.o,$^) $(LDFLAGS) $(LDLIBS) -o $@

$(BIN_DIR)/inkling_alloc_test.o: src/core/inkling_alloc.c src/core/inkling_alloc.h | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Dmalloc=inkling_test_malloc -c $< -o $@

$(BIN_DIR)/model_cli_probe.o: src/cli/inkling_run.c $(HEADERS) | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Dinkling_model_open=inkling_model_open_text_fixture -c $< -o $@

$(BIN_DIR):
	mkdir -p $@

$(SAFETENSORS_FIXTURE): tools/make_tiny_fixture.py
	python3 tools/make_tiny_fixture.py

test: $(BIN) $(TESTS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	$(BIN_DIR)/test_safetensors $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	$(BIN_DIR)/test_config $(CONFIG)
	$(BIN_DIR)/test_json $(CONFIG) $(INDEX)
	$(BIN_DIR)/test_catalogue $(CHECKPOINT_FIXTURES)
	$(BIN_DIR)/test_alloc
	$(BIN_DIR)/test_bf16
	$(BIN_DIR)/test_nvfp4 tests/fixtures/nvfp4.json
	python3 -B tests/test_nvfp4_fixture.py
	python3 -B tests/test_primitive_fixture.py
	$(BIN_DIR)/test_math tests/fixtures/primitives.json
	python3 tests/test_verify.py $(BIN) $(CHECKPOINT_FIXTURES)
	$(BIN_DIR)/test_model_bind tests/fixtures/model
	$(BIN_DIR)/test_model_runtime tests/fixtures/model
	$(BIN_DIR)/test_generate tests/fixtures/model
	python3 -B tests/test_generation_cli.py $(BIN) $(BIN_DIR)/model_cli_probe $(BIN_DIR)/generation_probe
	python3 -B tests/test_generation_fixture.py
	python3 -B tools/check_model.py --probe $(BIN_DIR)/model_probe --fixtures tests/fixtures/model
	python3 -B tests/test_model_fixture.py
	python3 -B tests/test_model_guards.py $(BIN_DIR)/model_probe $(BIN) $(BIN_DIR)/model_cli_probe

ORACLE_PYTHON ?= python3
ORACLE_FLAGS ?=

oracle-fixtures:
	$(ORACLE_PYTHON) -B tools/emit_fixtures.py $(ORACLE_FLAGS)

model-fixtures:
	$(ORACLE_PYTHON) -B tools/emit_model_fixtures.py $(ORACLE_FLAGS)

generation-fixtures:
	$(ORACLE_PYTHON) -B tools/emit_generation_fixtures.py $(ORACLE_FLAGS)

test-model-real: $(BIN_DIR)/model_probe
	python3 -B tools/check_model.py --probe $(BIN_DIR)/model_probe \
		--real weights/c16-real/layer-0 --real weights/c16-real/layer-2 \
		--real weights/c16-real/layer-5 --metadata $(CHECKPOINT_FIXTURES)

test-mutations:
	python3 -B tools/check_primitive_mutations.py --cc "$(CC)"
	python3 -B tools/check_model_mutations.py --cc "$(CC)"

test-sanitize:
	UBSAN_OPTIONS=halt_on_error=1 $(MAKE) test BIN_DIR=bin/sanitize \
		CFLAGS="$(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
		LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined"

clean:
	rm -f $(BIN) $(TESTS) $(BIN_DIR)/inkling_alloc_test.o $(BIN_DIR)/model_cli_probe.o
