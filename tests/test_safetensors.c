#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "inkling/inkling.h"

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

    uint64_t header_size = 0;

    if (!inkling_safetensors_header_size(
            argv[1],
            &header_size)) {
        return EXIT_FAILURE;
    }

    if (header_size != 64) {
        fprintf(
            stderr,
            "expected 64-byte header, got %" PRIu64 "\n",
            header_size
        );
        return EXIT_FAILURE;
    }

    printf(
        "Safetensors header OK: %" PRIu64 " bytes\n",
        header_size
    );

    return EXIT_SUCCESS;
}