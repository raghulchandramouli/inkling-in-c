CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror

BIN := bin/inkling
TEST_SAFETENSORS := bin/test_safetensors

CONFIG := tests/fixtures/inkling-small-config.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := tests/fixtures/inkling-shard-09-header.safetensors
ENCODER_FIXTURE := tests/fixtures/encoders.safetensors
NVFP4_FIXTURE := tests/fixtures/nvfp4.safetensors

INKLING_SOURCES := \
	src/cli/inkling_run.c \
	src/io/inkling_config.c

SAFETENSORS_TEST_SOURCES := \
	tests/test_safetensors.c \
	src/io/inkling_safetensors.c

.PHONY: all test clean

all: $(BIN)

$(BIN): $(INKLING_SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INKLING_SOURCES) -o $(BIN)

$(TEST_SAFETENSORS): $(SAFETENSORS_TEST_SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SAFETENSORS_TEST_SOURCES) -o $(TEST_SAFETENSORS) -lm

$(SAFETENSORS_FIXTURE): tools/make_tiny_fixture.py
	python3 tools/make_tiny_fixture.py

$(ENCODER_FIXTURE): tools/make_encoder_fixture.py
	python3 tools/make_encoder_fixture.py

$(NVFP4_FIXTURE): tools/make_nvfp4_fixture.py
	python3 tools/make_nvfp4_fixture.py

test: $(BIN) $(TEST_SAFETENSORS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER) $(ENCODER_FIXTURE) $(NVFP4_FIXTURE)
	./$(BIN) $(CONFIG)
	./$(TEST_SAFETENSORS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER) $(ENCODER_FIXTURE) $(NVFP4_FIXTURE)

clean:
	rm -f $(BIN) $(TEST_SAFETENSORS)
