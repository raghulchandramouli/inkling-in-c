CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror

BIN := bin/inkling
TESTS := bin/test_safetensors bin/test_config bin/test_json bin/test_catalogue
HEADERS := include/inkling/inkling.h src/io/inkling_json.h
IO_SOURCES := $(wildcard src/io/*.c)

CHECKPOINT_FIXTURES := tests/fixtures/checkpoint
CONFIG := $(CHECKPOINT_FIXTURES)/config.json
INDEX := $(CHECKPOINT_FIXTURES)/model.safetensors.index.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := $(CHECKPOINT_FIXTURES)/headers/model-00005-of-00009.safetensors

.PHONY: all test clean
all: $(BIN)

$(BIN): src/cli/inkling_run.c $(IO_SOURCES)
bin/test_safetensors: tests/test_safetensors.c src/io/inkling_safetensors.c src/io/inkling_json.c
bin/test_config: tests/test_config.c src/io/inkling_config.c src/io/inkling_json.c
bin/test_json: tests/test_json.c src/io/inkling_json.c
bin/test_catalogue: tests/test_catalogue.c src/io/inkling_index.c src/io/inkling_safetensors.c src/io/inkling_json.c

$(BIN) $(TESTS): $(HEADERS) | bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(filter %.c,$^) $(LDFLAGS) $(LDLIBS) -o $@

bin:
	mkdir -p $@

$(SAFETENSORS_FIXTURE): tools/make_tiny_fixture.py
	python3 tools/make_tiny_fixture.py

test: $(BIN) $(TESTS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./bin/test_safetensors $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./bin/test_config $(CONFIG)
	./bin/test_json $(CONFIG) $(INDEX)
	./bin/test_catalogue $(CHECKPOINT_FIXTURES)
	python3 tests/test_verify.py $(BIN) $(CHECKPOINT_FIXTURES)

clean:
	rm -f $(BIN) $(TESTS)
