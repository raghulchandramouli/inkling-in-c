#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

#define INDEX_MAX_BYTES (16L * 1024L * 1024L)
#define INDEX_INITIAL_CAPACITY 256

static char *read_text_file(const char *path)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(stderr, "cannot open shard index: %s\n", path);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }

    long length = ftell(file);

    if (length < 0 || length > INDEX_MAX_BYTES) {
        fprintf(stderr, "invalid shard index file size\n");
        fclose(file);
        return NULL;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    char *text = malloc((size_t)length + 1);

    if (text == NULL) {
        fprintf(stderr, "cannot allocate shard index buffer\n");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(text, 1, (size_t)length, file);
    fclose(file);

    if (bytes_read != (size_t)length) {
        fprintf(stderr, "could not read complete shard index\n");
        free(text);
        return NULL;
    }

    text[length] = '\0';
    return text;
}

static const char *skip_json_space(const char *position)
{
    while (isspace((unsigned char)*position)) {
        position++;
    }

    return position;
}

static const char *find_key(const char *json, const char *key)
{
    char pattern[64];

    int pattern_length = snprintf(
        pattern,
        sizeof(pattern),
        "\"%s\"",
        key
    );

    if (pattern_length < 0 ||
        (size_t)pattern_length >= sizeof(pattern)) {
        return NULL;
    }

    const char *position = strstr(json, pattern);

    if (position == NULL) {
        return NULL;
    }

    position += pattern_length;
    position = skip_json_space(position);

    if (*position != ':') {
        return NULL;
    }

    return skip_json_space(position + 1);
}

static int parse_u64_value(
    const char *json,
    const char *key,
    uint64_t *output
)
{
    const char *start = find_key(json, key);

    if (start == NULL || *start == '-') {
        return 0;
    }

    errno = 0;

    char *end = NULL;
    unsigned long long value = strtoull(start, &end, 10);

    if (errno == ERANGE || end == start) {
        return 0;
    }

    while (isspace((unsigned char)*end)) {
        end++;
    }

    if (*end != ',' && *end != '}') {
        return 0;
    }

    *output = (uint64_t)value;
    return 1;
}

static int read_quoted(
    const char **position,
    char *output,
    size_t output_size
)
{
    const char *start = skip_json_space(*position);

    if (*start != '"') {
        return 0;
    }

    const char *close = strchr(start + 1, '"');

    if (close == NULL) {
        return 0;
    }

    size_t length = (size_t)(close - start - 1);

    if (length == 0 || length >= output_size) {
        return 0;
    }

    if (memchr(start + 1, '"', length) != NULL ||
        memchr(start + 1, '\\', length) != NULL) {
        return 0;
    }

    memcpy(output, start + 1, length);
    output[length] = '\0';
    *position = close + 1;
    return 1;
}

static int reserve(
    InklingIndex *index,
    uint64_t needed
)
{
    if (needed <= index->capacity) {
        return 1;
    }

    uint64_t next_capacity = index->capacity == 0
        ? INDEX_INITIAL_CAPACITY
        : index->capacity;

    while (next_capacity < needed) {
        if (next_capacity > UINT64_MAX / 2) {
            return 0;
        }

        next_capacity *= 2;
    }

    if (next_capacity > SIZE_MAX / sizeof(InklingIndexEntry)) {
        return 0;
    }

    InklingIndexEntry *entries = realloc(
        index->entries,
        (size_t)next_capacity * sizeof(InklingIndexEntry)
    );

    if (entries == NULL) {
        fputs("cannot grow shard index\n", stderr);
        return 0;
    }

    index->entries = entries;
    index->capacity = next_capacity;
    return 1;
}

static int parse_weight_map(
    const char *json,
    InklingIndex *index
)
{
    const char *position = find_key(json, "weight_map");

    if (position == NULL || *position != '{') {
        fputs("shard index has no weight_map\n", stderr);
        return 0;
    }

    position++;

    for (;;) {
        position = skip_json_space(position);

        if (*position == '}') {
            return 1;
        }

        if (*position != '"') {
            fputs("malformed weight_map entry\n", stderr);
            return 0;
        }

        char name[INKLING_INDEX_MAX_NAME_LENGTH];
        char shard[INKLING_INDEX_MAX_SHARD_LENGTH];

        if (!read_quoted(&position, name, sizeof(name)) ||
            *skip_json_space(position) != ':') {
            fputs("malformed weight_map tensor name\n", stderr);
            return 0;
        }

        position = skip_json_space(position) + 1;

        if (!read_quoted(&position, shard, sizeof(shard))) {
            fputs("malformed weight_map shard name\n", stderr);
            return 0;
        }

        position = skip_json_space(position);

        if (!reserve(index, index->count + 1)) {
            return 0;
        }

        strcpy(index->entries[index->count].name, name);
        strcpy(index->entries[index->count].shard, shard);
        index->count++;

        if (*position == ',') {
            position++;
            continue;
        }

        if (*position == '}') {
            return 1;
        }

        fputs("malformed weight_map separator\n", stderr);
        return 0;
    }
}

int inkling_index_load(
    const char *path,
    InklingIndex *index
)
{
    if (path == NULL || index == NULL) {
        return 0;
    }

    char *json = read_text_file(path);

    if (json == NULL) {
        return 0;
    }

    InklingIndex parsed = {0};

    if (!parse_weight_map(json, &parsed) ||
        !parse_u64_value(
            json,
            "total_size",
            &parsed.total_size)) {
        fputs("invalid shard index\n", stderr);
        free(json);
        inkling_index_free(&parsed);
        return 0;
    }

    free(json);
    *index = parsed;
    return 1;
}

int inkling_index_find_shard(
    const InklingIndex *index,
    const char *tensor_name,
    char *shard_out,
    size_t shard_out_size
)
{
    if (index == NULL ||
        tensor_name == NULL ||
        shard_out == NULL) {
        return 0;
    }

    for (uint64_t entry = 0; entry < index->count; entry++) {
        if (strcmp(index->entries[entry].name, tensor_name) == 0) {
            size_t length = strlen(index->entries[entry].shard);

            if (length >= shard_out_size) {
                return 0;
            }

            memcpy(shard_out, index->entries[entry].shard, length + 1);
            return 1;
        }
    }

    return 0;
}

void inkling_index_free(InklingIndex *index)
{
    if (index == NULL) {
        return;
    }

    free(index->entries);
    index->entries = NULL;
    index->count = 0;
    index->capacity = 0;
    index->total_size = 0;
}