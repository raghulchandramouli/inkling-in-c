CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror

BIN := bin/inkling
TEST_SAFETENSORS := bin/test_safetensors

CONFIG := tests/fixtures/inkling-small-config.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := tests/fixtures/inkling-shard-09-header.safetensors

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
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SAFETENSORS_TEST_SOURCES) -o $(TEST_SAFETENSORS)

$(SAFETENSORS_FIXTURE): tools/make_tiny_fixture.py
	python3 tools/make_tiny_fixture.py

test: $(BIN) $(TEST_SAFETENSORS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./$(BIN) $(CONFIG)
	./$(TEST_SAFETENSORS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)

clean:
	rm -f $(BIN) $(TEST_SAFETENSORS)
