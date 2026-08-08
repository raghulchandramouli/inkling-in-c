#include <stdio.h>

#include "inkling/inkling.h"

#define SAFETENSORS_PREFIX_SIZE 8
#define SAFETENSORS_MAX_HEADER_SIZE (100ULL * 1024ULL * 1024ULL)

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

    unsigned char prefix[SAFETENSORS_PREFIX_SIZE];

    size_t bytes_read = fread(
        prefix,
        1,
        sizeof(prefix),
        file
    );

    fclose(file);

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