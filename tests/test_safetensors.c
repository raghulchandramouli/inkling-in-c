#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s <tiny.safetensors> <real-header.safetensors>\n",
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

    unsigned char tensor_bytes[4];

    if (!inkling_safetensors_read_tensor_data(
            argv[1],
            header_size,
            &tensor,
            tensor_bytes,
            sizeof(tensor_bytes))) {
        free(header);
        return EXIT_FAILURE;
    }

    uint32_t bits =
        (uint32_t)tensor_bytes[0] |
        (uint32_t)tensor_bytes[1] << 8 |
        (uint32_t)tensor_bytes[2] << 16 |
        (uint32_t)tensor_bytes[3] << 24;

    float value = 0.0f;

    if (sizeof(value) != sizeof(bits)) {
        fputs("this platform does not use 32-bit float\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    memcpy(&value, &bits, sizeof(value));

    if (value != 1.5f) {
        fprintf(stderr, "expected 1.5, got %.9g\n", value);
        free(header);
        return EXIT_FAILURE;
    }

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

    printf(
        "raw bytes:    %02x %02x %02x %02x\n",
        tensor_bytes[0],
        tensor_bytes[1],
        tensor_bytes[2],
        tensor_bytes[3]
    );

    printf("decoded F32:  %.1f\n", value);

    free(header);

    header = NULL;
    header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[2],
            &header,
            &header_size)) {
        return EXIT_FAILURE;
    }

    if (header_size != 5816) {
        fprintf(
            stderr,
            "expected 5816-byte real header, got %" PRIu64 "\n",
            header_size
        );
        free(header);
        return EXIT_FAILURE;
    }

    if (!inkling_safetensors_find_tensor(
            header,
            "model.llm.layers.0.attn.k_norm.weight",
            &tensor)) {
        fputs("could not parse real Inkling tensor\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    if (tensor.dtype != INKLING_DTYPE_BF16 ||
        tensor.rank != 1 ||
        tensor.shape[0] != 128 ||
        tensor.data_start != 3084 ||
        tensor.data_end != 3340) {
        fputs("incorrect real Inkling tensor metadata\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    uint64_t payload_size =
        tensor.data_end - tensor.data_start;

    file_offset = 8 + header_size + tensor.data_start;

    puts("Real Inkling metadata OK");
    puts("name:         model.llm.layers.0.attn.k_norm.weight");
    puts("dtype:        BF16");

    printf("shape:        [%" PRIu64 "]\n", tensor.shape[0]);
    printf("payload size: %" PRIu64 " bytes\n", payload_size);
    printf("file offset:  %" PRIu64 "\n", file_offset);

    free(header);
    return EXIT_SUCCESS;
}
