#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inkling_model_internal.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); exit(1); } } while (0)

static void compare_models(InklingModel *native, InklingModel *packed)
{
    unsigned char expected_bytes[1024], actual_bytes[1024];
    for (uint64_t i = 0; i < native->index.count; i++) {
        const InklingTensor *tensor = &native->index.entries[i];
        if (tensor->dtype == INKLING_DTYPE_UNKNOWN) continue;
        InklingRange range = {tensor->shard_id, tensor->data_offset, tensor->byte_length};
        for (uint64_t start = 0; start < range.bytes;) {
            size_t take = range.bytes - start > sizeof(expected_bytes) ? sizeof(expected_bytes) : (size_t)(range.bytes - start);
            CHECK(inkling_model_read(native, range, start, take, expected_bytes));
            CHECK(inkling_model_read(packed, range, start, take, actual_bytes));
            CHECK(memcmp(expected_bytes, actual_bytes, take) == 0);
            start += take;
        }
    }
    const size_t budgets[] = {96, 192, 1048576};
    uint32_t prompt[] = {2, 3, 7, 39};
    size_t width = inkling_model_logits_width(native), count = 4 * width;
    float *expected = malloc(count * sizeof(float)), *actual = malloc(count * sizeof(float));
    CHECK(expected && actual);
    for (size_t budget = 0; budget < sizeof(budgets) / sizeof(budgets[0]); budget++) {
        for (int incremental = 0; incremental <= 1; incremental++) {
            InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
            options.max_tokens = 12; options.weight_buffer_bytes = budgets[budget]; options.incremental = incremental;
            InklingContext *first = NULL, *second = NULL;
            CHECK(inkling_context_create_with_options(native, &options, &first));
            CHECK(inkling_context_create_with_options(packed, &options, &second));
            CHECK(inkling_forward(first, prompt, 4, expected, count));
            CHECK(inkling_forward(second, prompt, 4, actual, count));
            CHECK(memcmp(expected, actual, count * sizeof(float)) == 0);
            InklingGenerateOptions generate = INKLING_GENERATE_OPTIONS_INIT;
            generate.max_new_tokens = 5;
            uint32_t expected_ids[5], actual_ids[5];
            InklingGenerateResult expected_result, actual_result;
            CHECK(inkling_generate(first, prompt, 4, &generate, expected_ids, 5, &expected_result));
            CHECK(inkling_generate(second, prompt, 4, &generate, actual_ids, 5, &actual_result));
            CHECK(expected_result.generated_count == actual_result.generated_count);
            CHECK(expected_result.stop_reason == actual_result.stop_reason);
            CHECK(memcmp(expected_ids, actual_ids, expected_result.generated_count * sizeof(*expected_ids)) == 0);
            inkling_context_free(first); inkling_context_free(second);
        }
    }
    free(expected); free(actual);
}

int main(int argc, char **argv)
{
    if (argc != 4) return 2;
    InklingModelOptions options = {argv[3]};
    InklingModel *packed = NULL, *native = NULL;
    if (!inkling_model_open_text_fixture_with_options(argv[2], &options, &packed)) return 1;
    if (strcmp(argv[1], "--open") == 0) { inkling_model_close(packed); return 0; }
    CHECK(inkling_model_open_text_fixture(argv[2], &native));
    if (strcmp(argv[1], "--compare") == 0) compare_models(native, packed);
    else if (strcmp(argv[1], "--replace") == 0) {
        size_t length = strlen(argv[3]) + 8;
        char *held = malloc(length);
        CHECK(held);
        snprintf(held, length, "%s.held", argv[3]);
        CHECK(rename(argv[3], held) == 0);
        FILE *file = fopen(argv[3], "wb");
        CHECK(file && fputs("replacement", file) >= 0 && fclose(file) == 0);
        unsigned char expected[64], actual[64];
        CHECK(inkling_model_read(native, native->embed.data, 0, sizeof(expected), expected));
        CHECK(inkling_model_read(packed, packed->embed.data, 0, sizeof(actual), actual));
        CHECK(memcmp(expected, actual, sizeof(expected)) == 0);
        CHECK(truncate(held, 0) == 0);
        CHECK(!inkling_model_read(packed, packed->embed.data, 0, sizeof(actual), actual));
        free(held);
    } else return 2;
    inkling_model_close(native); inkling_model_close(packed);
    return 0;
}
