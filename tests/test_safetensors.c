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

    char *header = NULL;
    uint64_t header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[1],
            &header,
            &header_size)) {
        return EXIT_FAILURE;
    }

    InklingTensorInfo tensor;

    if (!inkling_safetensors_find_tensor(
            header,
            "test.weight",
            &tensor)) {
        fputs("could not parse test.weight\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    if (tensor.dtype != INKLING_DTYPE_F32 ||
        tensor.rank != 1 ||
        tensor.shape[0] != 1 ||
        tensor.data_start != 0 ||
        tensor.data_end != 4) {
        fputs("incorrect tensor metadata\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    uint64_t file_offset =
        8 + header_size + tensor.data_start;

    puts("Tensor metadata OK");
    puts("name:         test.weight");
    puts("dtype:        F32");

    printf("rank:         %" PRIu32 "\n", tensor.rank);
    printf("shape:        [%" PRIu64 "]\n", tensor.shape[0]);

    printf(
        "data offsets: [%" PRIu64 ", %" PRIu64 ")\n",
        tensor.data_start,
        tensor.data_end
    );

    printf("file offset:  %" PRIu64 "\n", file_offset);

    free(header);
    return EXIT_SUCCESS;
}
