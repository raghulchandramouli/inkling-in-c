#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

static const char expected_header[] =
    "{\"test.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
    "\"data_offsets\":[0,4]}}";

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(
            stderr,
            "usage: %s <file.safetensors>\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    char *header = NULL;
    uint64_t header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[1],
            &header,
            &header_size)) {
        return EXIT_FAILURE;
    }

    if (header_size != (uint64_t)strlen(expected_header)) {
        fprintf(
            stderr,
            "unexpected header size: %" PRIu64 "\n",
            header_size
        );
        free(header);
        return EXIT_FAILURE;
    }

    if (strcmp(header, expected_header) != 0) {
        fputs("Safetensors header content does not match\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    printf(
        "Safetensors JSON header OK: %" PRIu64 " bytes\n",
        header_size
    );

    printf("%s\n", header);

    free(header);
    return EXIT_SUCCESS;
}