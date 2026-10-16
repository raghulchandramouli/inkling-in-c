#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model/inkling_model_internal.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    InklingModel *model = NULL;
    InklingContext *first = NULL, *second = NULL, *invalid = NULL;
    CHECK(inkling_model_open_text_fixture(argv[1], &model));
    CHECK(inkling_context_create(model, 3, &first));
    CHECK(inkling_context_create(model, 3, &second));
    CHECK(!inkling_context_create(model, SIZE_MAX, &invalid));
    CHECK(invalid == NULL);
    CHECK(!inkling_context_create(model, 0, &invalid));
    CHECK(invalid == NULL);
    size_t width = inkling_model_logits_width(model), count = 3 * width;
    CHECK(width < model->config.vocab_size);
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    CHECK(expected != NULL && actual != NULL);
    uint32_t ids[3] = {2, 3, model->config.vocab_size - 1};
    CHECK(inkling_forward(first, ids, 3, expected, count));
    CHECK(inkling_forward(second, ids, 3, actual, count));
    CHECK(memcmp(expected, actual, count * sizeof(float)) == 0);
    inkling_context_reset(first);
    CHECK(inkling_forward(first, ids, 3, actual, count));
    CHECK(memcmp(expected, actual, count * sizeof(float)) == 0);
    CHECK(!inkling_forward(first, ids, 3, actual, count - 1));
    CHECK(!inkling_forward(first, ids, 0, actual, count));
    CHECK(!inkling_forward(first, ids, 4, actual, count));
    CHECK(!inkling_forward(first, NULL, 3, actual, count));
    CHECK(!inkling_forward(first, ids, 3, NULL, count));
    CHECK(!inkling_forward(first, (const uint32_t *)actual, 3, actual, count));
    uint32_t bad_ids[3] = {2, model->config.vocab_size, 3};
    CHECK(!inkling_forward(first, bad_ids, 3, actual, count));
    CHECK(inkling_forward(first, ids, 3, actual, count));
    CHECK(memcmp(expected, actual, count * sizeof(float)) == 0);
    CHECK(inkling_forward(first, ids, 1, actual, count));
    CHECK(memcmp(expected, actual, width * sizeof(float)) == 0);
    size_t hidden_count = 3 * model->config.hidden_size;
    float *input = calloc(hidden_count + 1, sizeof(float));
    float *output = malloc(hidden_count * sizeof(float));
    CHECK(input != NULL && output != NULL);
    for (size_t i = 0; i < hidden_count; i++) input[i] = (float)(i % 7) / 13.0f;
    const InklingLayer *layer = &model->layers[0];
    CHECK(!inkling_attention_forward(first, layer, input, 3, 0, input));
    CHECK(!inkling_mlp_forward(first, layer, input, 3, input + 1));
    CHECK(!inkling_layer_forward(first, layer, input, 3, input));
    CHECK(!inkling_attention_forward(first, layer, input, 3, SIZE_MAX, output));
    input[0] = NAN;
    CHECK(!inkling_layer_forward(first, layer, input, 3, output));
    input[0] = 0;
    CHECK(inkling_layer_forward(first, layer, input, 3, output));
    for (size_t i = 0; i < hidden_count; i++) CHECK(isfinite(output[i]));
    CHECK(inkling_forward(first, ids, 3, actual, count));
    CHECK(memcmp(expected, actual, count * sizeof(float)) == 0);
    InklingModel incomplete = *model;
    incomplete.complete = 0;
    CHECK(inkling_context_create(&incomplete, 3, &invalid));
    CHECK(!inkling_forward(invalid, ids, 3, actual, count));
    inkling_context_free(invalid);
    inkling_context_free(first);
    inkling_context_free(second);
    inkling_context_reset(NULL);
    inkling_context_free(NULL);
    inkling_model_close(model);
    free(expected); free(actual); free(input); free(output);
    puts("model lifecycle checks passed");
    return 0;
}
