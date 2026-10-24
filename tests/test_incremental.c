#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling_model_internal.h"
#include "inkling_json.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

static InklingContext *make_context(const InklingModel *model, size_t capacity, int incremental)
{
    InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
    options.max_tokens = capacity;
    options.incremental = incremental;
    options.expert_cache_bytes = 128 * 1024;
    InklingContext *context = NULL;
    CHECK(inkling_context_create_with_options(model, &options, &context));
    return context;
}

static void equal_row(const float *expected, const float *actual, size_t width, size_t position)
{
    for (size_t i = 0; i < width; i++) {
        if (memcmp(expected + i, actual + i, sizeof(float)) != 0) {
            fprintf(stderr, "incremental mismatch position %zu logit %zu: %.9g != %.9g\n",
                    position, i, (double)expected[i], (double)actual[i]);
            exit(1);
        }
    }
}

static void compare_sequence(InklingModel *model, size_t tokens)
{
    size_t width = inkling_model_logits_width(model);
    uint32_t *ids = malloc(tokens * sizeof(*ids));
    float *expected = malloc(tokens * width * sizeof(*expected));
    float *actual = malloc(width * sizeof(*actual));
    float *other = malloc(4 * width * sizeof(*other));
    const uint32_t other_ids[] = {19, 1, 4, 2};
    CHECK(ids != NULL && expected != NULL && actual != NULL && other != NULL);
    for (size_t i = 0; i < tokens; i++) ids[i] = (uint32_t)((i * 7 + 2) % model->config.vocab_size);
    InklingContext *reference = make_context(model, tokens, 0);
    InklingContext *first = make_context(model, tokens, 1);
    InklingContext *second = make_context(model, tokens, 1);
    CHECK(first->scratch_capacity == 1);
    CHECK(inkling_forward(reference, ids, tokens, expected, tokens * width));
    CHECK(inkling_forward(reference, other_ids, 4, other, 4 * width));
    CHECK(inkling_prefill(first, ids, 1, actual, width));
    equal_row(expected, actual, width, 0);
    for (size_t i = 1; i < tokens; i++) {
        CHECK(inkling_decode(first, ids[i], actual, width));
        CHECK(inkling_context_position(first) == i + 1);
        equal_row(expected + i * width, actual, width, i);
        if (i == 3) {
            CHECK(inkling_prefill(second, other_ids, 3, actual, width));
            equal_row(other + 2 * width, actual, width, 2);
            CHECK(inkling_decode(second, other_ids[3], actual, width));
            equal_row(other + 3 * width, actual, width, 3);
        }
    }
    CHECK(first->scratch_capacity == 1);
    CHECK(inkling_prefill(second, ids, tokens - 1, actual, width));
    equal_row(expected + (tokens - 2) * width, actual, width, tokens - 2);
    CHECK(inkling_decode(second, ids[tokens - 1], actual, width));
    equal_row(expected + (tokens - 1) * width, actual, width, tokens - 1);
    CHECK(!inkling_decode(first, ids[0], actual, width));
    CHECK(inkling_context_position(first) == 0);
    CHECK(!inkling_decode(first, ids[0], actual, width));
    CHECK(inkling_prefill(first, ids, tokens, actual, width));
    equal_row(expected + (tokens - 1) * width, actual, width, tokens - 1);
    InklingRuntimeStats stats;
    inkling_context_get_stats(first, &stats);
    CHECK(stats.tokens_processed == 2 * tokens);
    CHECK(stats.bytes_read > 0 && stats.read_calls > 0);
    inkling_context_free(reference);
    inkling_context_free(first);
    inkling_context_free(second);
    free(ids); free(expected); free(actual); free(other);
}

static void failures_and_reset(InklingModel *model)
{
    const uint32_t ids[] = {2, 3, 7, 9, 4, 8};
    const size_t tokens = sizeof(ids) / sizeof(ids[0]);
    size_t width = inkling_model_logits_width(model);
    float *expected = malloc(tokens * width * sizeof(*expected));
    float *actual = malloc(width * sizeof(*actual));
    float *sentinel = malloc(width * sizeof(*sentinel));
    CHECK(expected != NULL && actual != NULL && sentinel != NULL);
    for (size_t i = 0; i < width; i++) sentinel[i] = (float)i + 100;
    InklingContext *reference = make_context(model, tokens, 0);
    InklingContext *context = make_context(model, tokens, 1);
    CHECK(inkling_forward(reference, ids, tokens, expected, tokens * width));
    memcpy(actual, sentinel, width * sizeof(float));
    CHECK(!inkling_prefill(context, ids, 0, actual, width));
    CHECK(memcmp(actual, sentinel, width * sizeof(float)) == 0);
    CHECK(!inkling_decode(context, ids[0], actual, width));
    CHECK(!inkling_prefill(context, ids, tokens, actual, width - 1));
    CHECK(memcmp(actual, sentinel, width * sizeof(float)) == 0);
    CHECK(!inkling_prefill(context, (const uint32_t *)actual, 1, actual, width));
    uint32_t invalid[] = {2, model->config.vocab_size};
    CHECK(!inkling_prefill(context, invalid, 2, actual, width));
    CHECK(memcmp(actual, sentinel, width * sizeof(float)) == 0);
    CHECK(inkling_prefill(context, ids, 3, actual, width));
    CHECK(!inkling_decode(context, model->config.vocab_size, actual, width));
    CHECK(inkling_context_position(context) == 0);
    CHECK(!inkling_decode(context, ids[3], actual, width));
    inkling_context_reset(context);
    CHECK(inkling_decode(context, ids[0], actual, width));
    equal_row(expected, actual, width, 0);
    CHECK(inkling_prefill(context, ids, 3, actual, width));
    float saved = model->layers[0].attn_norm[0];
    model->layers[0].attn_norm[0] = NAN;
    CHECK(!inkling_decode(context, ids[3], actual, width));
    model->layers[0].attn_norm[0] = saved;
    CHECK(inkling_context_position(context) == 0);
    CHECK(!inkling_decode(context, ids[3], actual, width));
    CHECK(inkling_prefill(context, ids, 3, actual, width));
    CHECK(inkling_decode(context, ids[3], actual, width));
    equal_row(expected + 3 * width, actual, width, 3);
    CHECK(inkling_forward(context, ids, tokens, expected, tokens * width));
    CHECK(context->scratch_capacity == tokens);
    CHECK(inkling_prefill(context, ids, tokens, actual, width));
    equal_row(expected + (tokens - 1) * width, actual, width, tokens - 1);
    CHECK(inkling_context_position(NULL) == 0);
    CHECK(!inkling_prefill(NULL, ids, 1, actual, width));
    CHECK(!inkling_decode(NULL, ids[0], actual, width));
    InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
    InklingContext *invalid_context = (InklingContext *)1;
    CHECK(!inkling_context_create_with_options(model, NULL, &invalid_context));
    CHECK(invalid_context == NULL);
    CHECK(!inkling_context_create_with_options(model, &options, &invalid_context));
    options.max_tokens = tokens;
    options.incremental = 2;
    CHECK(!inkling_context_create_with_options(model, &options, &invalid_context));
    options.incremental = 1;
    options.weight_buffer_bytes = 1;
    CHECK(!inkling_context_create_with_options(model, &options, &invalid_context));
    options.weight_buffer_bytes = 2 * model->config.hidden_size;
    CHECK(!inkling_context_create_with_options(model, &options, &invalid_context));
    inkling_context_free(context);
    inkling_context_free(reference);
    free(expected); free(actual); free(sentinel);
}

static void oracle_prefixes(InklingModel *model, const char *directory)
{
    char path[4096];
    CHECK(snprintf(path, sizeof(path), "%s/oracle.json", directory) > 0);
    InklingJsonDocument document = {0};
    CHECK(inkling_json_load_file(path, 64L * 1024L * 1024L, &document));
    const InklingJsonValue *cases = inkling_json_object_get(document.root, "cases");
    size_t width = inkling_model_logits_width(model);
    float *actual = malloc(width * sizeof(*actual));
    CHECK(actual != NULL);
    size_t checked = 0;
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        const InklingJsonValue *test = inkling_json_array_at(cases, c);
        const char *kind = inkling_json_string(inkling_json_object_get(test, "kind"), NULL);
        if (kind == NULL || strcmp(kind, "model") != 0) continue;
        const InklingJsonValue *input = inkling_json_object_get(test, "input_ids");
        const InklingJsonValue *expected = inkling_json_object_get(test, "expected");
        size_t tokens = inkling_json_array_size(input);
        uint32_t *ids = malloc(tokens * sizeof(*ids));
        CHECK(ids != NULL && inkling_json_array_size(expected) == tokens * width);
        for (size_t i = 0; i < tokens; i++) {
            uint64_t id;
            CHECK(inkling_json_number_u64(inkling_json_array_at(input, i), &id));
            CHECK(id <= UINT32_MAX);
            ids[i] = (uint32_t)id;
        }
        InklingContext *context = make_context(model, tokens, 1);
        for (size_t i = 0; i < tokens; i++) {
            CHECK(i == 0 ? inkling_prefill(context, ids, 1, actual, width)
                         : inkling_decode(context, ids[i], actual, width));
            for (size_t j = 0; j < width; j++) {
                double value;
                CHECK(inkling_json_number_double(inkling_json_array_at(expected, i * width + j), &value));
                CHECK(fabs((double)actual[j] - value) <= 0.00001 + 0.0005 * fabs(value));
            }
        }
        checked++;
        inkling_context_free(context);
        free(ids);
    }
    CHECK(checked == 3);
    free(actual);
    inkling_json_document_free(&document);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    InklingModel *model = NULL;
    CHECK(inkling_model_open_text_fixture(argv[1], &model));
    oracle_prefixes(model, argv[1]);
    failures_and_reset(model);
    compare_sequence(model, 521);
    model->config.log_scaling_n_floor = 2;
    model->config.relative_extent = 3;
    model->config.sliding_window_size = 4;
    for (size_t i = 0; i < model->layer_count; i++) {
        model->layers[i].extent = 3;
        if (model->layers[i].window != 0) model->layers[i].window = 4;
    }
    compare_sequence(model, 17);
    inkling_model_close(model);
    puts("incremental sequence checks passed");
    return 0;
}
