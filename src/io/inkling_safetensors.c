#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

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