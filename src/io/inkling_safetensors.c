#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

#define SAFETENSORS_PREFIX_SIZE 8
#define SAFETENSORS_MAX_HEADER_SIZE (100ULL * 1024ULL * 1024ULL)

static int read_header_size(
    FILE *file,
    uint64_t *header_size
)
{
    unsigned char prefix[SAFETENSORS_PREFIX_SIZE];

    size_t bytes_read = fread(
        prefix,
        1,
        sizeof(prefix),
        file
    );

    if (bytes_read != sizeof(prefix)) {
        fputs("Safetensors file is shorter than 8 bytes\n", stderr);
        return 0;
    }

    uint64_t length = 0;

    for (unsigned int index = 0;
         index < SAFETENSORS_PREFIX_SIZE;
         index++) {
        length |= (uint64_t)prefix[index] << (index * 8);
    }

    if (length == 0 ||
        length > SAFETENSORS_MAX_HEADER_SIZE) {
        fputs("invalid Safetensors header size\n", stderr);
        return 0;
    }

    *header_size = length;
    return 1;
}

int inkling_safetensors_header_size(
    const char *path,
    uint64_t *header_size
)
{
    if (path == NULL || header_size == NULL) {
        return 0;
    }

    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(stderr, "cannot open Safetensors file: %s\n", path);
        return 0;
    }

    int success = read_header_size(file, header_size);

    fclose(file);
    return success;
}

int inkling_safetensors_read_header(
    const char *path,
    char **header_json,
    uint64_t *header_size
)
{
    if (path == NULL ||
        header_json == NULL ||
        header_size == NULL) {
        return 0;
    }

    *header_json = NULL;
    *header_size = 0;

    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(stderr, "cannot open Safetensors file: %s\n", path);
        return 0;
    }

    uint64_t length = 0;

    if (!read_header_size(file, &length)) {
        fclose(file);
        return 0;
    }

    char *json = malloc((size_t)length + 1);

    if (json == NULL) {
        fputs("cannot allocate Safetensors header\n", stderr);
        fclose(file);
        return 0;
    }

    size_t bytes_read = fread(
        json,
        1,
        (size_t)length,
        file
    );

    fclose(file);

    if (bytes_read != (size_t)length) {
        fputs("incomplete Safetensors header\n", stderr);
        free(json);
        return 0;
    }

    json[(size_t)length] = '\0';

    size_t first_character = 0;

    while (first_character < (size_t)length &&
           isspace((unsigned char)json[first_character])) {
        first_character++;
    }

    if (first_character == (size_t)length ||
        json[first_character] != '{') {
        fputs("Safetensors header is not JSON\n", stderr);
        free(json);
        return 0;
    }

    *header_json = json;
    *header_size = length;

    return 1;
}

static const char *skip_json_space(
    const char *position,
    const char *end
)
{
    while (position < end &&
           isspace((unsigned char)*position)) {
        position++;
    }

    return position;
}

static const char *find_object_value(
    const char *object_start,
    const char *object_end,
    const char *key
)
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

    const char *position = strstr(object_start, pattern);

    if (position == NULL || position >= object_end) {
        return NULL;
    }

    position += pattern_length;
    position = skip_json_space(position, object_end);

    if (position >= object_end || *position != ':') {
        return NULL;
    }

    position++;
    return skip_json_space(position, object_end);
}

static int parse_u64(
    const char **position,
    const char *end,
    uint64_t *output
)
{
    const char *start = skip_json_space(*position, end);

    if (start >= end ||
        !isdigit((unsigned char)*start)) {
        return 0;
    }

    errno = 0;

    char *number_end = NULL;
    unsigned long long value = strtoull(
        start,
        &number_end,
        10
    );

    if (errno == ERANGE ||
        number_end == start ||
        number_end > end) {
        return 0;
    }

    *output = (uint64_t)value;
    *position = number_end;

    return 1;
}

static int parse_dtype(
    const char *object_start,
    const char *object_end,
    InklingDataType *dtype
)
{
    const char *value = find_object_value(
        object_start,
        object_end,
        "dtype"
    );

    if (value == NULL) {
        return 0;
    }

    size_t remaining = (size_t)(object_end - value);

    if (remaining >= 5 &&
        memcmp(value, "\"F32\"", 5) == 0) {
        *dtype = INKLING_DTYPE_F32;
        return 1;
    }

    if (remaining >= 6 &&
        memcmp(value, "\"BF16\"", 6) == 0) {
        *dtype = INKLING_DTYPE_BF16;
        return 1;
    }

    return 0;
}

static int parse_shape(
    const char *object_start,
    const char *object_end,
    InklingTensorInfo *tensor
)
{
    const char *position = find_object_value(
        object_start,
        object_end,
        "shape"
    );

    if (position == NULL || *position != '[') {
        return 0;
    }

    position++;
    position = skip_json_space(position, object_end);

    tensor->rank = 0;

    if (position < object_end && *position == ']') {
        return 1;
    }

    while (position < object_end) {
        if (tensor->rank >= INKLING_MAX_TENSOR_RANK) {
            return 0;
        }

        if (!parse_u64(
                &position,
                object_end,
                &tensor->shape[tensor->rank])) {
            return 0;
        }

        tensor->rank++;
        position = skip_json_space(position, object_end);

        if (position >= object_end) {
            return 0;
        }

        if (*position == ']') {
            return 1;
        }

        if (*position != ',') {
            return 0;
        }

        position++;
    }

    return 0;
}

static int parse_offsets(
    const char *object_start,
    const char *object_end,
    InklingTensorInfo *tensor
)
{
    const char *position = find_object_value(
        object_start,
        object_end,
        "data_offsets"
    );

    if (position == NULL || *position != '[') {
        return 0;
    }

    position++;

    if (!parse_u64(
            &position,
            object_end,
            &tensor->data_start)) {
        return 0;
    }

    position = skip_json_space(position, object_end);

    if (position >= object_end || *position != ',') {
        return 0;
    }

    position++;

    if (!parse_u64(
            &position,
            object_end,
            &tensor->data_end)) {
        return 0;
    }

    position = skip_json_space(position, object_end);

    return position < object_end && *position == ']';
}

static uint64_t dtype_size(InklingDataType dtype)
{
    switch (dtype) {
    case INKLING_DTYPE_F32:
        return 4;

    case INKLING_DTYPE_BF16:
        return 2;

    default:
        return 0;
    }
}

static int tensor_size_is_valid(
    const InklingTensorInfo *tensor
)
{
    uint64_t element_size = dtype_size(tensor->dtype);

    if (element_size == 0 ||
        tensor->data_end < tensor->data_start) {
        return 0;
    }

    uint64_t element_count = 1;

    for (uint32_t dimension = 0;
         dimension < tensor->rank;
         dimension++) {
        uint64_t size = tensor->shape[dimension];

        if (size != 0 &&
            element_count > UINT64_MAX / size) {
            return 0;
        }

        element_count *= size;
    }

    if (element_count > UINT64_MAX / element_size) {
        return 0;
    }

    uint64_t expected_bytes =
        element_count * element_size;

    uint64_t actual_bytes =
        tensor->data_end - tensor->data_start;

    return expected_bytes == actual_bytes;
}

int inkling_safetensors_find_tensor(
    const char *header_json,
    const char *tensor_name,
    InklingTensorInfo *tensor
)
{
    if (header_json == NULL ||
        tensor_name == NULL ||
        tensor == NULL) {
        return 0;
    }

    if (strchr(tensor_name, '"') != NULL ||
        strchr(tensor_name, '\\') != NULL) {
        return 0;
    }

    char name_pattern[512];

    int name_length = snprintf(
        name_pattern,
        sizeof(name_pattern),
        "\"%s\"",
        tensor_name
    );

    if (name_length < 0 ||
        (size_t)name_length >= sizeof(name_pattern)) {
        return 0;
    }

    const char *name_position = strstr(
        header_json,
        name_pattern
    );

    if (name_position == NULL) {
        return 0;
    }

    const char *header_end =
        header_json + strlen(header_json);

    const char *position =
        name_position + name_length;

    position = skip_json_space(position, header_end);

    if (position >= header_end || *position != ':') {
        return 0;
    }

    position++;
    position = skip_json_space(position, header_end);

    if (position >= header_end || *position != '{') {
        return 0;
    }

    const char *object_start = position + 1;
    const char *object_end = strchr(object_start, '}');

    if (object_end == NULL) {
        return 0;
    }

    InklingTensorInfo parsed = {0};

    if (!parse_dtype(
            object_start,
            object_end,
            &parsed.dtype) ||
        !parse_shape(
            object_start,
            object_end,
            &parsed) ||
        !parse_offsets(
            object_start,
            object_end,
            &parsed) ||
        !tensor_size_is_valid(&parsed)) {
        return 0;
    }

    *tensor = parsed;
    return 1;
}

int inkling_safetensors_read_tensor_data(
    const char *path,
    uint64_t header_size,
    const InklingTensorInfo *tensor,
    void *destination,
    size_t destination_size
)
{
    if (path == NULL ||
        tensor == NULL ||
        destination == NULL ||
        tensor->data_end < tensor->data_start) {
        return 0;
    }

    uint64_t tensor_size =
        tensor->data_end - tensor->data_start;

    if (tensor_size > SIZE_MAX ||
        destination_size < (size_t)tensor_size) {
        fputs("tensor destination is too small\n", stderr);
        return 0;
    }

    if (header_size > UINT64_MAX - SAFETENSORS_PREFIX_SIZE ||
        tensor->data_start >
            UINT64_MAX - SAFETENSORS_PREFIX_SIZE - header_size) {
        fputs("Safetensors tensor offset overflow\n", stderr);
        return 0;
    }

    uint64_t file_offset =
        SAFETENSORS_PREFIX_SIZE +
        header_size +
        tensor->data_start;

    if (file_offset > LONG_MAX) {
        fputs("Safetensors tensor offset is unsupported\n", stderr);
        return 0;
    }

    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(stderr, "cannot open Safetensors file: %s\n", path);
        return 0;
    }

    if (fseek(file, (long)file_offset, SEEK_SET) != 0) {
        fputs("cannot seek to Safetensors tensor\n", stderr);
        fclose(file);
        return 0;
    }

    size_t bytes_read = fread(
        destination,
        1,
        (size_t)tensor_size,
        file
    );

    fclose(file);

    if (bytes_read != (size_t)tensor_size) {
        fputs("incomplete Safetensors tensor data\n", stderr);
        return 0;
    }

    return 1;
}
