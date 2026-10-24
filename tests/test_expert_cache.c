#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling_model_internal.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

static int fail_cache_allocation;

void *inkling_test_cache_malloc(size_t bytes)
{
    if (fail_cache_allocation) {
        fail_cache_allocation = 0;
        return NULL;
    }
    return malloc(bytes);
}

typedef struct {
    InklingCacheEvent events[64];
    size_t count;
} CacheTrace;

static void cache_trace(void *user, const InklingCacheEvent *event)
{
    CacheTrace *trace = user;
    CHECK(trace->count < sizeof(trace->events) / sizeof(trace->events[0]));
    trace->events[trace->count++] = *event;
}

static InklingRuntimeStats stats_of(InklingContext *context)
{
    InklingRuntimeStats stats;
    inkling_context_get_stats(context, &stats);
    CHECK(stats.cache_bytes <= stats.cache_capacity);
    CHECK(stats.cache_peak_bytes <= stats.cache_capacity);
    CHECK(stats.cache_bytes <= stats.cache_peak_bytes);
    return stats;
}

static InklingContext *context_for(const InklingModel *model, size_t cache,
                                    size_t buffer, int incremental)
{
    InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
    options.max_tokens = 16;
    options.expert_cache_bytes = cache;
    options.weight_buffer_bytes = buffer;
    options.incremental = incremental;
    InklingContext *context = NULL;
    CHECK(inkling_context_create_with_options(model, &options, &context));
    CHECK(context != NULL);
    InklingRuntimeStats stats = stats_of(context);
    CHECK(stats.cache_capacity == cache && stats.cache_bytes == 0);
    CHECK(stats.cache_peak_bytes == 0 && stats.bytes_read == 0 && stats.read_calls == 0);
    return context;
}

static size_t row_bytes(const InklingMatrix *matrix)
{
    if (matrix->rows == 0) return 0;
    return matrix->kind == INKLING_MATRIX_BF16 ? matrix->columns * 2 :
        matrix->columns / 2 + matrix->columns / 16;
}

static size_t minimum_buffer(const InklingModel *model)
{
    size_t bytes = row_bytes(&model->embed);
    if (row_bytes(&model->unembed) > bytes) bytes = row_bytes(&model->unembed);
    for (size_t i = 0; i < model->layer_count; i++) {
        const InklingLayer *layer = &model->layers[i];
        const InklingMatrix *matrices[] = {&layer->q, &layer->k, &layer->v,
            &layer->r, &layer->o, &layer->w13, &layer->down,
            &layer->shared_w13, &layer->shared_down, &layer->router};
        for (size_t j = 0; j < sizeof(matrices) / sizeof(matrices[0]); j++)
            if (row_bytes(matrices[j]) > bytes) bytes = row_bytes(matrices[j]);
    }
    return bytes;
}

static void projection_input(const InklingMatrix *matrix, float *input)
{
    for (size_t i = 0; i < matrix->columns; i++)
        input[i] = (float)((int)(i % 13) - 6) / 17.0f;
}

static void project(InklingContext *context, const InklingMatrix *matrix,
                     size_t expert, float *output)
{
    CHECK(matrix->columns <= 128 && matrix->rows <= 128);
    float input[128];
    projection_input(matrix, input);
    CHECK(inkling_project(context, matrix, expert, matrix->rows, input, output));
    for (size_t i = 0; i < matrix->rows; i++) CHECK(isfinite(output[i]));
}

static void event_is(const CacheTrace *trace, size_t index,
                       InklingCacheEventKind kind, size_t layer, size_t expert,
                       size_t projection, size_t bytes, size_t resident)
{
    CHECK(index < trace->count);
    const InklingCacheEvent *event = &trace->events[index];
    CHECK(event->kind == kind && event->layer == layer && event->expert == expert);
    CHECK(event->projection == projection && event->bytes == bytes);
    CHECK(event->resident_bytes == resident);
}

static size_t discover_charge(InklingModel *model, const InklingMatrix *matrix,
                                size_t buffer)
{
    InklingContext *context = context_for(model, 1048576, buffer, 1);
    float output[128];
    project(context, matrix, 0, output);
    InklingRuntimeStats stats = stats_of(context);
    size_t payload = matrix->rows * row_bytes(matrix);
    CHECK(stats.cache_bytes > payload);
    CHECK(stats.cache_peak_bytes == stats.cache_bytes);
    CHECK(stats.bytes_read == payload && stats.read_calls == 2);
    CHECK(stats.cache_misses == 1 && stats.cache_hits == 0 && stats.cache_evictions == 0);
    size_t charge = stats.cache_bytes;
    inkling_context_free(context);
    return charge;
}

static void capacity_and_lru(InklingModel *model, const InklingLayer *layer,
                               size_t charge, size_t buffer)
{
    const InklingMatrix *matrix = &layer->w13;
    const size_t access[] = {0, 1, 0, 2, 1};
    const size_t capacities[] = {0, 1, charge - 1, charge, 2 * charge};
    float expected[5][128], actual[128];
    InklingContext *stream = context_for(model, 0, buffer, 1);
    for (size_t i = 0; i < 5; i++) project(stream, matrix, access[i], expected[i]);
    InklingRuntimeStats streaming = stats_of(stream);
    CHECK(streaming.bytes_read == 5 * matrix->rows * row_bytes(matrix));
    inkling_context_free(stream);
    for (size_t c = 0; c < sizeof(capacities) / sizeof(capacities[0]); c++) {
        size_t capacity = capacities[c];
        InklingContext *context = context_for(model, capacity, buffer, 1);
        CacheTrace trace = {{{0}}, 0};
        inkling_context_cache_trace(context, cache_trace, &trace);
        size_t event = 0;
        for (size_t i = 0; i < 5; i++) {
            size_t before = capacity < charge ? 0 :
                capacity == charge || i < 2 ? (i == 0 ? 0 : charge) : 2 * charge;
            int hit = capacity == 2 * charge && i == 2;
            project(context, matrix, access[i], actual);
            CHECK(memcmp(expected[i], actual, matrix->rows * sizeof(float)) == 0);
            if (hit) {
                event_is(&trace, event++, INKLING_CACHE_HIT, layer->original_id,
                         access[i], 0, charge, before);
            } else {
                event_is(&trace, event++, INKLING_CACHE_MISS, layer->original_id,
                         access[i], 0, 0, before);
                if (capacity < charge) {
                    event_is(&trace, event++, INKLING_CACHE_BYPASS, layer->original_id,
                             access[i], 0, charge, 0);
                } else if ((capacity == charge && i != 0) ||
                           (capacity == 2 * charge && i >= 3)) {
                    size_t evicted = capacity == charge ? access[i - 1] :
                        i == 3 ? 1 : 0;
                    event_is(&trace, event++, INKLING_CACHE_EVICT, layer->original_id,
                             evicted, 0, charge, before - charge);
                }
            }
            InklingRuntimeStats stats = stats_of(context);
            size_t hits = capacity == 2 * charge && i >= 2 ? 1 : 0;
            size_t misses = i + 1 - hits;
            size_t resident = capacity < charge ? 0 :
                capacity == charge || i == 0 ? charge : 2 * charge;
            size_t evictions = capacity < charge ? 0 :
                capacity == charge ? i : i >= 3 ? i - 2 : 0;
            CHECK(stats.cache_hits == hits && stats.cache_misses == misses);
            CHECK(stats.cache_evictions == evictions && stats.cache_bytes == resident);
            CHECK(stats.cache_peak_bytes == resident && trace.count == event);
            CHECK(stats.bytes_read == misses * matrix->rows * row_bytes(matrix));
            if (capacity >= charge) CHECK(stats.read_calls == 2 * misses);
        }
        InklingRuntimeStats stats = stats_of(context);
        if (capacity == 2 * charge) {
            CHECK(stats.bytes_read < streaming.bytes_read);
            CHECK(stats.read_calls < streaming.read_calls);
        }
        inkling_context_free(context);
    }
}

static void key_identity(InklingModel *model, InklingLayer *layer,
                           size_t charge, size_t down_charge, size_t buffer)
{
    InklingLayer *other = NULL;
    for (size_t i = 0; i < model->layer_count; i++)
        if (&model->layers[i] != layer && !model->layers[i].dense)
            other = &model->layers[i];
    CHECK(other != NULL);
    InklingMatrix saved = other->w13;
    other->w13 = layer->w13;
    InklingContext *context = context_for(model, 2 * charge + down_charge, buffer, 1);
    CacheTrace trace = {{{0}}, 0};
    inkling_context_cache_trace(context, cache_trace, &trace);
    float first[128], second[128], output[128];
    project(context, &layer->w13, 1, first);
    project(context, &other->w13, 1, second);
    CHECK(memcmp(first, second, layer->w13.rows * sizeof(float)) == 0);
    project(context, &layer->down, 1, output);
    project(context, &layer->w13, 1, second);
    CHECK(memcmp(first, second, layer->w13.rows * sizeof(float)) == 0);
    project(context, &other->w13, 1, second);
    project(context, &layer->down, 1, second);
    CHECK(memcmp(output, second, layer->down.rows * sizeof(float)) == 0);
    CHECK(trace.count == 6);
    event_is(&trace, 0, INKLING_CACHE_MISS, layer->original_id, 1, 0, 0, 0);
    event_is(&trace, 1, INKLING_CACHE_MISS, other->original_id, 1, 0, 0, charge);
    event_is(&trace, 2, INKLING_CACHE_MISS, layer->original_id, 1, 1, 0, 2 * charge);
    event_is(&trace, 3, INKLING_CACHE_HIT, layer->original_id, 1, 0, charge, 2 * charge + down_charge);
    event_is(&trace, 4, INKLING_CACHE_HIT, other->original_id, 1, 0, charge, 2 * charge + down_charge);
    event_is(&trace, 5, INKLING_CACHE_HIT, layer->original_id, 1, 1, down_charge, 2 * charge + down_charge);
    InklingRuntimeStats stats = stats_of(context);
    CHECK(stats.cache_hits == 3 && stats.cache_misses == 3 && stats.cache_evictions == 0);
    inkling_context_free(context);
    other->w13 = saved;
}

static void multiple_evictions(InklingModel *model, InklingLayer *layer,
                                 size_t charge, size_t down_charge, size_t buffer)
{
    CHECK(charge > down_charge && charge <= 2 * down_charge);
    InklingContext *context = context_for(model, 2 * down_charge, buffer, 1);
    float output[128];
    project(context, &layer->down, 0, output);
    project(context, &layer->down, 1, output);
    CacheTrace trace = {{{0}}, 0};
    inkling_context_cache_trace(context, cache_trace, &trace);
    project(context, &layer->w13, 0, output);
    CHECK(trace.count == 3);
    event_is(&trace, 0, INKLING_CACHE_MISS, layer->original_id, 0, 0, 0, 2 * down_charge);
    event_is(&trace, 1, INKLING_CACHE_EVICT, layer->original_id, 0, 1, down_charge, down_charge);
    event_is(&trace, 2, INKLING_CACHE_EVICT, layer->original_id, 1, 1, down_charge, 0);
    InklingRuntimeStats stats = stats_of(context);
    CHECK(stats.cache_evictions == 2 && stats.cache_bytes == charge);
    CHECK(stats.cache_peak_bytes == 2 * down_charge);
    inkling_context_free(context);
}

static void failed_entries(InklingModel *model, InklingLayer *layer,
                             size_t charge, size_t buffer)
{
    InklingMatrix *matrix = &layer->w13;
    float input[128], output[128], expected[128];
    projection_input(matrix, input);
    InklingContext *stream = context_for(model, 0, buffer, 1);
    project(stream, matrix, 1, expected);
    inkling_context_free(stream);
    for (size_t i = 0; i < 2; i++) {
        InklingContext *context = context_for(model, charge, buffer, 1);
        project(context, matrix, 0, output);
        InklingRange *range = i == 0 ? &matrix->data : &matrix->scales;
        InklingRange saved = *range;
        range->bytes = 0;
        CHECK(!inkling_project(context, matrix, 1, matrix->rows, input, output));
        *range = saved;
        InklingRuntimeStats stats = stats_of(context);
        CHECK(stats.cache_bytes == 0 && stats.cache_peak_bytes == charge);
        CHECK(stats.cache_hits == 0 && stats.cache_misses == 2 && stats.cache_evictions == 1);
        project(context, matrix, 1, output);
        CHECK(memcmp(expected, output, matrix->rows * sizeof(float)) == 0);
        stats = stats_of(context);
        CHECK(stats.cache_hits == 0 && stats.cache_misses == 3 && stats.cache_bytes == charge);
        project(context, matrix, 1, output);
        CHECK(stats_of(context).cache_hits == 1);
        inkling_context_free(context);
    }
    InklingContext *context = context_for(model, charge, buffer, 1);
    project(context, matrix, 0, output);
    CacheTrace trace = {{{0}}, 0};
    inkling_context_cache_trace(context, cache_trace, &trace);
    fail_cache_allocation = 1;
    project(context, matrix, 1, output);
    CHECK(fail_cache_allocation == 0);
    CHECK(memcmp(expected, output, matrix->rows * sizeof(float)) == 0);
    CHECK(trace.count == 3);
    event_is(&trace, 0, INKLING_CACHE_MISS, layer->original_id, 1, 0, 0, charge);
    event_is(&trace, 1, INKLING_CACHE_EVICT, layer->original_id, 0, 0, charge, 0);
    event_is(&trace, 2, INKLING_CACHE_BYPASS, layer->original_id, 1, 0, charge, 0);
    InklingRuntimeStats stats = stats_of(context);
    CHECK(stats.cache_bytes == 0 && stats.cache_peak_bytes == charge);
    CHECK(stats.cache_hits == 0 && stats.cache_misses == 2 && stats.cache_evictions == 1);
    inkling_context_cache_trace(context, NULL, NULL);
    project(context, matrix, 1, output);
    CHECK(stats_of(context).cache_hits == 0 && stats_of(context).cache_misses == 3);
    inkling_context_free(context);
}

static void independent_contexts(InklingModel *model, InklingLayer *layer,
                                   size_t charge, size_t buffer)
{
    InklingContext *first = context_for(model, charge, buffer, 1);
    InklingContext *second = context_for(model, charge, buffer, 1);
    float a[128], b[128];
    project(first, &layer->w13, 0, a);
    CHECK(stats_of(second).cache_bytes == 0 && stats_of(second).cache_misses == 0);
    project(second, &layer->w13, 1, b);
    InklingRuntimeStats before = stats_of(second);
    inkling_context_reset(first);
    CHECK(stats_of(first).cache_bytes == charge);
    inkling_context_reset_stats(first);
    InklingRuntimeStats stats = stats_of(first);
    CHECK(stats.cache_capacity == charge && stats.cache_bytes == charge && stats.cache_peak_bytes == charge);
    CHECK(stats.bytes_read == 0 && stats.read_calls == 0 && stats.cache_hits == 0);
    CHECK(stats.cache_misses == 0 && stats.cache_evictions == 0 && stats.tokens_processed == 0);
    project(first, &layer->w13, 0, b);
    CHECK(memcmp(a, b, layer->w13.rows * sizeof(float)) == 0);
    stats = stats_of(first);
    CHECK(stats.cache_hits == 1 && stats.cache_misses == 0 && stats.read_calls == 0);
    stats = stats_of(second);
    CHECK(memcmp(&before, &stats, sizeof(stats)) == 0);
    inkling_context_free(first);
    project(second, &layer->w13, 1, b);
    CHECK(stats_of(second).cache_hits == 1 && stats_of(second).cache_misses == 1);
    inkling_context_free(second);
}

static void saturated_stats(InklingModel *model, InklingLayer *layer,
                              size_t charge, size_t buffer)
{
    InklingContext *context = context_for(model, charge, buffer, 1);
    context->stats.bytes_read = UINT64_MAX - 1;
    context->stats.read_calls = UINT64_MAX - 1;
    context->stats.cache_misses = UINT64_MAX - 1;
    context->stats.cache_hits = UINT64_MAX - 1;
    context->stats.cache_evictions = UINT64_MAX - 1;
    float output[128];
    project(context, &layer->w13, 0, output);
    project(context, &layer->w13, 0, output);
    project(context, &layer->w13, 0, output);
    project(context, &layer->w13, 1, output);
    project(context, &layer->w13, 2, output);
    InklingRuntimeStats stats = stats_of(context);
    CHECK(stats.bytes_read == UINT64_MAX && stats.read_calls == UINT64_MAX);
    CHECK(stats.cache_misses == UINT64_MAX && stats.cache_hits == UINT64_MAX);
    CHECK(stats.cache_evictions == UINT64_MAX);
    size_t width = inkling_model_logits_width(model);
    uint32_t ids[] = {2, 5};
    float logits[128];
    CHECK(2 * width <= 128);
    context->stats.tokens_processed = UINT64_MAX - 1;
    CHECK(inkling_forward(context, ids, 2, logits, 128));
    CHECK(stats_of(context).tokens_processed == UINT64_MAX);
    CHECK(inkling_forward(context, ids, 2, logits, 128));
    CHECK(stats_of(context).tokens_processed == UINT64_MAX);
    CHECK(inkling_prefill(context, ids, 2, logits, 128));
    CHECK(stats_of(context).tokens_processed == UINT64_MAX);
    inkling_context_free(context);
}

static void generation_parity(InklingModel *model, size_t charge, size_t buffer)
{
    const size_t capacities[] = {0, charge, 1048576};
    const size_t buffers[] = {buffer, 3 * buffer, 1048576};
    uint32_t prompts[4][4] = {{1, 5, 7, 9}, {39, 3, 12, 8}, {6, 14, 2, 17}, {3, 31, 21, 15}};
    InklingContext *reference = context_for(model, 0, 1048576, 0);
    size_t longer_runs = 0;
    for (size_t p = 0; p < 4; p++) {
        for (size_t sampled = 0; sampled < 2; sampled++) {
            InklingGenerateOptions options = INKLING_GENERATE_OPTIONS_INIT;
            options.max_new_tokens = 8;
            if (sampled) {
                options.temperature = 0.7;
                options.top_k = 9;
                options.top_p = 0.8;
                options.seed = UINT64_C(0x123456789abcdef) + p;
            }
            InklingGenerateResult expected;
            uint32_t ids[8];
            inkling_context_reset_stats(reference);
            CHECK(inkling_generate(reference, prompts[p], p + 1, &options, ids, 8, &expected));
            InklingRuntimeStats full_stats = stats_of(reference);
            size_t generated = expected.generated_count;
            CHECK(generated > 0);
            CHECK(full_stats.tokens_processed == generated * (p + 1) + generated * (generated - 1) / 2);
            for (size_t incremental = 0; incremental < 2; incremental++) {
                for (size_t c = 0; c < sizeof(capacities) / sizeof(capacities[0]); c++) {
                    for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); b++) {
                        InklingContext *context = context_for(model, capacities[c], buffers[b], (int)incremental);
                        uint32_t actual[8];
                        InklingGenerateResult result;
                        CHECK(inkling_generate(context, prompts[p], p + 1, &options, actual, 8, &result));
                        CHECK(result.generated_count == generated && result.stop_reason == expected.stop_reason);
                        CHECK(memcmp(ids, actual, generated * sizeof(*ids)) == 0);
                        InklingRuntimeStats stats = stats_of(context);
                        if (incremental) {
                            CHECK(stats.tokens_processed == p + generated);
                            if (generated > 1) {
                                CHECK(stats.tokens_processed < full_stats.tokens_processed);
                                longer_runs++;
                            }
                        } else CHECK(stats.tokens_processed == full_stats.tokens_processed);
                        inkling_context_free(context);
                    }
                }
            }
        }
    }
    CHECK(longer_runs > 0);
    inkling_context_free(reference);
}

static void logits_parity(InklingModel *model, size_t charge, size_t buffer)
{
    const size_t capacities[] = {0, charge, 1048576};
    const size_t buffers[] = {buffer, 3 * buffer, 1048576};
    const uint32_t prompt[] = {2, 5, 7, 39, 8, 9};
    size_t width = inkling_model_logits_width(model), count = 6 * width;
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    CHECK(expected != NULL && actual != NULL);
    InklingContext *reference = context_for(model, 0, 1048576, 0);
    CHECK(inkling_forward(reference, prompt, 6, expected, count));
    inkling_context_free(reference);
    for (size_t c = 0; c < sizeof(capacities) / sizeof(capacities[0]); c++) {
        for (size_t b = 0; b < sizeof(buffers) / sizeof(buffers[0]); b++) {
            InklingContext *context = context_for(model, capacities[c], buffers[b], 1);
            CHECK(inkling_forward(context, prompt, 6, actual, count));
            CHECK(memcmp(expected, actual, count * sizeof(float)) == 0);
            CHECK(inkling_prefill(context, prompt, 3, actual, width));
            for (size_t t = 2; t < 6; t++) {
                if (t > 2) CHECK(inkling_decode(context, prompt[t], actual, width));
                for (size_t j = 0; j < width; j++)
                    CHECK(fabsf(expected[t * width + j] - actual[j]) <= 0.00002f);
            }
            inkling_context_free(context);
        }
    }
    free(expected);
    free(actual);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    InklingModel *model = NULL;
    CHECK(inkling_model_open_text_fixture(argv[1], &model));
    InklingLayer *layer = NULL;
    for (size_t i = 0; i < model->layer_count; i++)
        if (!model->layers[i].dense && model->layers[i].w13.kind == INKLING_MATRIX_NVFP4)
            layer = &model->layers[i];
    CHECK(layer != NULL && layer->w13.experts >= 3);
    size_t buffer = minimum_buffer(model);
    CHECK(buffer > 1);
    InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
    options.max_tokens = 16;
    options.weight_buffer_bytes = buffer - 1;
    InklingContext *invalid = (InklingContext *)(void *)model;
    CHECK(!inkling_context_create_with_options(model, &options, &invalid));
    CHECK(invalid == NULL);
    size_t charge = discover_charge(model, &layer->w13, buffer);
    size_t down_charge = discover_charge(model, &layer->down, buffer);
    capacity_and_lru(model, layer, charge, buffer);
    key_identity(model, layer, charge, down_charge, buffer);
    multiple_evictions(model, layer, charge, down_charge, buffer);
    failed_entries(model, layer, charge, buffer);
    independent_contexts(model, layer, charge, buffer);
    saturated_stats(model, layer, charge, buffer);
    logits_parity(model, charge, buffer);
    generation_parity(model, charge, buffer);
    InklingRuntimeStats stats;
    memset(&stats, 0xa5, sizeof(stats));
    inkling_context_get_stats(NULL, &stats);
    CHECK(stats.bytes_read == 0 && stats.cache_bytes == 0 && stats.cache_capacity == 0);
    inkling_context_reset_stats(NULL);
    inkling_context_cache_trace(NULL, NULL, NULL);
    inkling_context_get_stats(NULL, NULL);
    inkling_model_close(model);
    puts("expert cache, bounded streaming, trace, and generation parity checks passed");
    return 0;
}
