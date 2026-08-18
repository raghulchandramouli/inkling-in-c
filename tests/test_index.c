#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

static int failures = 0;

static void expect(int condition, const char *what)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static void expect_shard(
    const InklingIndex *index,
    const char *tensor_name,
    const char *expected_shard
)
{
    char shard[INKLING_INDEX_MAX_SHARD_LENGTH];

    if (!inkling_index_find_shard(
            index,
            tensor_name,
            shard,
            sizeof(shard))) {
        fprintf(
            stderr,
            "FAIL: tensor %s not found in index\n",
            tensor_name
        );
        failures++;
        return;
    }

    if (strcmp(shard, expected_shard) != 0) {
        fprintf(
            stderr,
            "FAIL: %s maps to %s, expected %s\n",
            tensor_name,
            shard,
            expected_shard
        );
        failures++;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s <index.json> <real-header.safetensors>\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    InklingIndex index;

    if (!inkling_index_load(argv[1], &index)) {
        return EXIT_FAILURE;
    }

    if (index.count != 1048) {
        fprintf(
            stderr,
            "FAIL: expected 1048 tensors, got %" PRIu64 "\n",
            index.count
        );
        failures++;
    }

    if (index.total_size != 531912898740ULL) {
        fprintf(
            stderr,
            "FAIL: expected total_size 531912898740, got %" PRIu64 "\n",
            index.total_size
        );
        failures++;
    }

    expect_shard(
        &index,
        "model.llm.layers.0.attn.k_norm.weight",
        "model-00009-of-00032.safetensors"
    );

    expect_shard(
        &index,
        "model.llm.embed.weight",
        "model-00030-of-00032.safetensors"
    );

    expect_shard(
        &index,
        "model.visual.layers.linear_2.weight",
        "model-00009-of-00032.safetensors"
    );

    expect_shard(
        &index,
        "model.mtp.layers.0.transformer_block.attn.rel_logits_proj.proj",
        "mtp.safetensors"
    );

    char shard[INKLING_INDEX_MAX_SHARD_LENGTH];

    expect(
        !inkling_index_find_shard(
            &index,
            "no.such.tensor",
            shard,
            sizeof(shard)),
        "unknown tensor not found"
    );

    expect(
        !inkling_index_find_shard(
            &index,
            NULL,
            shard,
            sizeof(shard)),
        "NULL tensor name rejected"
    );

    puts("Shard index OK");
    printf("tensors:      %" PRIu64 "\n", index.count);
    printf("total size:   %" PRIu64 " bytes\n", index.total_size);

    if (failures != 0) {
        fprintf(stderr, "%d index test(s) failed\n", failures);
        inkling_index_free(&index);
        return EXIT_FAILURE;
    }

    /*
     * End to end: resolve a tensor through the index, then prove it
     * exists in the real shard header fixture (shard 09 of 32).
     */
    const char *resolved_name =
        "model.llm.layers.0.attn.k_norm.weight";

    if (!inkling_index_find_shard(
            &index,
            resolved_name,
            shard,
            sizeof(shard))) {
        fputs("could not resolve k_norm.weight shard\n", stderr);
        inkling_index_free(&index);
        return EXIT_FAILURE;
    }

    char *header = NULL;
    uint64_t header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[2],
            &header,
            &header_size)) {
        inkling_index_free(&index);
        return EXIT_FAILURE;
    }

    InklingTensorInfo tensor;

    if (!inkling_safetensors_find_tensor(
            header,
            resolved_name,
            &tensor)) {
        fputs(
            "resolved tensor missing from real shard header\n",
            stderr
        );
        free(header);
        inkling_index_free(&index);
        return EXIT_FAILURE;
    }

    if (tensor.dtype != INKLING_DTYPE_BF16 ||
        tensor.rank != 1 ||
        tensor.shape[0] != 128 ||
        tensor.data_start != 3084 ||
        tensor.data_end != 3340) {
        fputs("incorrect tensor metadata from real shard\n", stderr);
        free(header);
        inkling_index_free(&index);
        return EXIT_FAILURE;
    }

    printf("resolved:     %s -> %s\n", resolved_name, shard);
    printf("shard header: %" PRIu64 " bytes\n", header_size);

    free(header);
    inkling_index_free(&index);

    puts("Index end-to-end OK");
    return EXIT_SUCCESS;
}