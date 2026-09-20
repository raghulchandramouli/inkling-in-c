CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror

BIN := bin/inkling
TEST_SAFETENSORS := bin/test_safetensors
TEST_CONFIG := bin/test_config
TEST_INDEX := bin/test_index
TEST_JSON := bin/test_json

CHECKPOINT_FIXTURES := tests/fixtures/checkpoint
CONFIG := $(CHECKPOINT_FIXTURES)/config.json
INDEX := $(CHECKPOINT_FIXTURES)/model.safetensors.index.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := \
	$(CHECKPOINT_FIXTURES)/headers/model-00005-of-00009.safetensors

INKLING_SOURCES := \
	src/cli/inkling_run.c \
	src/io/inkling_config.c

SAFETENSORS_TEST_SOURCES := \
	tests/test_safetensors.c \
	src/io/inkling_safetensors.c

CONFIG_TEST_SOURCES := \
	tests/test_config.c \
	src/io/inkling_config.c

INDEX_TEST_SOURCES := \
	tests/test_index.c \
	src/io/inkling_index.c \
	src/io/inkling_safetensors.c

JSON_TEST_SOURCES := \
	tests/test_json.c \
	src/io/inkling_json.c

.PHONY: all test clean

all: $(BIN)

$(BIN): $(INKLING_SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INKLING_SOURCES) -o $(BIN)

$(TEST_SAFETENSORS): $(SAFETENSORS_TEST_SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SAFETENSORS_TEST_SOURCES) -o $(TEST_SAFETENSORS)

$(TEST_CONFIG): $(CONFIG_TEST_SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CONFIG_TEST_SOURCES) -o $(TEST_CONFIG)

$(TEST_INDEX): $(INDEX_TEST_SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INDEX_TEST_SOURCES) -o $(TEST_INDEX)

$(TEST_JSON): $(JSON_TEST_SOURCES) src/io/inkling_json.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(JSON_TEST_SOURCES) -o $(TEST_JSON)

$(SAFETENSORS_FIXTURE): tools/make_tiny_fixture.py
	python3 tools/make_tiny_fixture.py

test: $(BIN) $(TEST_SAFETENSORS) $(TEST_CONFIG) $(TEST_INDEX) $(TEST_JSON) \
	$(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./$(BIN) $(CONFIG)
	./$(TEST_SAFETENSORS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./$(TEST_CONFIG) $(CONFIG)
	./$(TEST_INDEX) $(INDEX) $(REAL_SAFETENSORS_HEADER)
	./$(TEST_JSON) $(CONFIG) $(INDEX)

clean:
	rm -f $(BIN) $(TEST_SAFETENSORS) $(TEST_CONFIG) $(TEST_INDEX) $(TEST_JSON)
