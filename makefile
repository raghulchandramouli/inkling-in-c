CC ?= cc

CPPFLAGS := -Iinclude
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror

BIN := bin/inkling
SOURCES := \
	src/cli/inkling_run.c \
	src/io/inkling_config.c

.PHONY: all test clean

all: $(BIN)

$(BIN): $(SOURCES) include/inkling/inkling.h
	mkdir -p bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SOURCES) -o $(BIN)

test: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)