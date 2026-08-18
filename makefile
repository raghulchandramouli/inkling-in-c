CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror

BIN := bin/inkling
TEST_SAFETENSORS := bin/test_safetensors
TEST_CONFIG := bin/test_config
TEST_INDEX := bin/test_index

CONFIG := tests/fixtures/inkling-small-config.json
INDEX := tests/fixtures/inkling-small-index.json
SAFETENSORS_FIXTURE := tests/fixtures/tiny.safetensors
REAL_SAFETENSORS_HEADER := tests/fixtures/inkling-shard-09-header.safetensors

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

$(SAFETENSORS_FIXTURE): tools/make_tiny_fixture.py
	python3 tools/make_tiny_fixture.py

test: $(BIN) $(TEST_SAFETENSORS) $(TEST_CONFIG) $(TEST_INDEX) \
	$(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./$(BIN) $(CONFIG)
	./$(TEST_SAFETENSORS) $(SAFETENSORS_FIXTURE) $(REAL_SAFETENSORS_HEADER)
	./$(TEST_CONFIG) $(CONFIG)
	./$(TEST_INDEX) $(INDEX) $(REAL_SAFETENSORS_HEADER)

clean:
	rm -f $(BIN) $(TEST_SAFETENSORS) $(TEST_CONFIG) $(TEST_INDEX)
