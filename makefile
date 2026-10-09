CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wpointer-arith -ffp-contract=off

BIN_DIR ?= bin
BIN := $(BIN_DIR)/inkling
TESTS := $(addprefix $(BIN_DIR)/,test_safetensors test_config test_json test_catalogue test_alloc test_bf16 test_nvfp4)
HEADERS := include/inkling/inkling.h src/io/inkling_json.h $(wildcard src/core/*.h)
IO_SOURCES := $(wildcard src/io/*.c)
CORE_SOURCES := $(wildcard src/core/*.c)

CHECKPOINT_FIXTURES := tests/fixtures/checkpoint
CONFIG := $(CHECKPOINT_FIXTURES)/config.json
INDEX := $(CHECKPOINT_FIXTURES)/model.safetensors.index.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := $(CHECKPOINT_FIXTURES)/headers/model-00005-of-00009.safetensors

.PHONY: all test test-sanitize clean
all: $(BIN)

$(BIN): src/cli/inkling_run.c $(IO_SOURCES) $(CORE_SOURCES)
$(BIN_DIR)/test_safetensors: tests/test_safetensors.c src/io/inkling_safetensors.c src/io/inkling_json.c
$(BIN_DIR)/test_config: tests/test_config.c src/io/inkling_config.c src/io/inkling_json.c
$(BIN_DIR)/test_json: tests/test_json.c src/io/inkling_json.c
$(BIN_DIR)/test_catalogue: tests/test_catalogue.c src/io/inkling_index.c src/io/inkling_safetensors.c src/io/inkling_json.c
$(BIN_DIR)/test_alloc: tests/test_alloc.c $(BIN_DIR)/inkling_alloc_test.o
$(BIN_DIR)/test_bf16: tests/test_bf16.c src/core/inkling_bf16.c
$(BIN_DIR)/test_nvfp4: tests/test_nvfp4.c src/core/inkling_nvfp4.c src/core/inkling_alloc.c src/core/inkling_bf16.c src/io/inkling_json.c

$(BIN) $(TESTS): $(HEADERS) | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(filter %.c %.o,$^) $(LDFLAGS) $(LDLIBS) -o $@

$(BIN_DIR)/inkling_alloc_test.o: src/core/inkling_alloc.c src/core/inkling_alloc.h | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Dmalloc=inkling_test_malloc -c $< -o $@

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
	python3 tests/test_verify.py $(BIN) $(CHECKPOINT_FIXTURES)

test-sanitize:
	UBSAN_OPTIONS=halt_on_error=1 $(MAKE) test BIN_DIR=bin/sanitize \
		CFLAGS="$(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
		LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined"

clean:
	rm -f $(BIN) $(TESTS) $(BIN_DIR)/inkling_alloc_test.o
