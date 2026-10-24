#include <stdlib.h>
#include <string.h>

#include "inkling_alloc.h"
#include "inkling_math.h"
#include "inkling_model_internal.h"

struct InklingExpertCache {
    struct InklingExpertCache *newer, *older;
    const InklingMatrix *matrix;
    size_t expert, charge, data_bytes;
    unsigned char payload[];
};

static void increment(uint64_t *counter, uint64_t amount)
{
    *counter = amount > UINT64_MAX - *counter ? UINT64_MAX : *counter + amount;
}

int inkling_context_read(InklingContext *context, InklingRange range, uint64_t relative,
                          size_t bytes, void *destination)
{
    increment(&context->stats.read_calls, 1);
    if (!inkling_model_read(context->model, range, relative, bytes, destination)) return 0;
    increment(&context->stats.bytes_read, bytes);
    return 1;
}

void inkling_context_get_stats(const InklingContext *context, InklingRuntimeStats *stats)
{
    if (stats != NULL) {
        if (context != NULL) *stats = context->stats;
        else memset(stats, 0, sizeof(*stats));
    }
}

void inkling_context_reset_stats(InklingContext *context)
{
    if (context == NULL) return;
    size_t used = context->stats.cache_bytes;
    memset(&context->stats, 0, sizeof(context->stats));
    context->stats.cache_bytes = used;
    context->stats.cache_peak_bytes = used;
    context->stats.cache_capacity = context->expert_cache_capacity;
}

void inkling_context_cache_trace(InklingContext *context, InklingCacheTrace trace, void *user)
{
    if (context == NULL) return;
    context->cache_trace = trace;
    context->cache_trace_user = user;
}

static void cache_event(InklingContext *context, const InklingMatrix *matrix,
                         size_t expert, InklingCacheEventKind kind, size_t bytes)
{
    if (context->cache_trace == NULL) return;
    for (size_t i = 0; i < context->model->layer_count; i++) {
        const InklingLayer *layer = &context->model->layers[i];
        if (matrix != &layer->w13 && matrix != &layer->down) continue;
        InklingCacheEvent event = {kind, layer->original_id, expert,
            matrix == &layer->w13 ? 0 : 1, bytes, context->stats.cache_bytes};
        context->cache_trace(context->cache_trace_user, &event);
        return;
    }
}

void inkling_expert_cache_free(struct InklingExpertCache *cache)
{
    while (cache != NULL) {
        struct InklingExpertCache *next = cache->older;
        free(cache);
        cache = next;
    }
}

static int routed_matrix(const InklingModel *model, const InklingMatrix *matrix)
{
    if (matrix->kind != INKLING_MATRIX_NVFP4) return 0;
    for (size_t i = 0; i < model->layer_count; i++) {
        const InklingLayer *layer = &model->layers[i];
        if (!layer->dense && (matrix == &layer->w13 || matrix == &layer->down)) return 1;
    }
    return 0;
}

static void promote(InklingContext *context, struct InklingExpertCache *entry)
{
    if (entry == context->expert_cache) return;
    if (entry->newer != NULL) entry->newer->older = entry->older;
    if (entry->older != NULL) entry->older->newer = entry->newer;
    entry->newer = NULL;
    entry->older = context->expert_cache;
    if (context->expert_cache != NULL) context->expert_cache->newer = entry;
    context->expert_cache = entry;
}

static int cached_expert(InklingContext *context, const InklingMatrix *matrix,
                          size_t expert, const unsigned char **data,
                          const unsigned char **scales)
{
    *data = NULL;
    *scales = NULL;
    if (!routed_matrix(context->model, matrix)) return 1;
    for (struct InklingExpertCache *entry = context->expert_cache; entry != NULL; entry = entry->older) {
        if (entry->matrix != matrix || entry->expert != expert) continue;
        promote(context, entry);
        increment(&context->stats.cache_hits, 1);
        cache_event(context, matrix, expert, INKLING_CACHE_HIT, entry->charge);
        *data = entry->payload;
        *scales = entry->payload + entry->data_bytes;
        return 1;
    }
    increment(&context->stats.cache_misses, 1);
    cache_event(context, matrix, expert, INKLING_CACHE_MISS, 0);
    size_t data_bytes, scale_bytes, payload_bytes, charge;
    if (!inkling_size_mul(matrix->rows, matrix->columns / 2, &data_bytes) ||
        !inkling_size_mul(matrix->rows, matrix->columns / 16, &scale_bytes) ||
        !inkling_size_add(data_bytes, scale_bytes, &payload_bytes) ||
        !inkling_size_add(sizeof(struct InklingExpertCache), payload_bytes, &charge)) return 0;
    if (charge > context->expert_cache_capacity || charge > (size_t)PTRDIFF_MAX) {
        cache_event(context, matrix, expert, INKLING_CACHE_BYPASS, charge);
        return 1;
    }
    while (context->stats.cache_bytes > context->expert_cache_capacity - charge) {
        struct InklingExpertCache *tail = context->expert_cache;
        while (tail->older != NULL) tail = tail->older;
        if (tail->newer != NULL) tail->newer->older = NULL;
        else context->expert_cache = NULL;
        context->stats.cache_bytes -= tail->charge;
        cache_event(context, tail->matrix, tail->expert, INKLING_CACHE_EVICT, tail->charge);
        free(tail);
        increment(&context->stats.cache_evictions, 1);
    }
    struct InklingExpertCache *entry = malloc(charge);
    if (entry == NULL) {
        cache_event(context, matrix, expert, INKLING_CACHE_BYPASS, charge);
        return 1;
    }
    size_t data_offset, scale_offset;
    if (!inkling_size_mul(expert, data_bytes, &data_offset) ||
        !inkling_size_mul(expert, scale_bytes, &scale_offset) ||
        !inkling_context_read(context, matrix->data, data_offset, data_bytes, entry->payload) ||
        !inkling_context_read(context, matrix->scales, scale_offset, scale_bytes, entry->payload + data_bytes)) {
        free(entry);
        return 0;
    }
    entry->matrix = matrix;
    entry->expert = expert;
    entry->charge = charge;
    entry->data_bytes = data_bytes;
    entry->newer = NULL;
    entry->older = context->expert_cache;
    if (context->expert_cache != NULL) context->expert_cache->newer = entry;
    context->expert_cache = entry;
    context->stats.cache_bytes += charge;
    if (context->stats.cache_bytes > context->stats.cache_peak_bytes)
        context->stats.cache_peak_bytes = context->stats.cache_bytes;
    *data = entry->payload;
    *scales = entry->payload + data_bytes;
    return 1;
}

int inkling_project(InklingContext *context, const InklingMatrix *matrix,
                    size_t expert, size_t rows, const float *input, float *output)
{
    size_t row_bytes, scale_bytes = 0, total_row_bytes, expert_row;
    if (expert >= matrix->experts || rows == 0 || rows > matrix->rows ||
        !inkling_size_mul(expert, matrix->rows, &expert_row)) return 0;
    if (matrix->kind == INKLING_MATRIX_BF16) {
        if (!inkling_size_mul(matrix->columns, 2, &row_bytes)) return 0;
    } else {
        row_bytes = matrix->columns / 2;
        scale_bytes = matrix->columns / 16;
    }
    if (!inkling_size_add(row_bytes, scale_bytes, &total_row_bytes) ||
        total_row_bytes == 0 || total_row_bytes > context->weight_bytes) return 0;
    const unsigned char *cached_data = NULL, *cached_scales = NULL;
    if (!cached_expert(context, matrix, expert, &cached_data, &cached_scales)) return 0;
    size_t tile_rows = context->weight_bytes / total_row_bytes;
    for (size_t row = 0; row < rows;) {
        size_t count = rows - row < tile_rows ? rows - row : tile_rows;
        size_t data_bytes = count * row_bytes;
        size_t source_row, offset;
        if (!inkling_size_add(expert_row, row, &source_row) ||
            !inkling_size_mul(source_row, row_bytes, &offset)) return 0;
        const unsigned char *data = context->weights, *scales = NULL;
        if (cached_data != NULL) {
            data = cached_data + row * row_bytes;
            scales = cached_scales + row * scale_bytes;
        } else if (!inkling_context_read(context, matrix->data, offset,
                                          data_bytes, context->weights)) return 0;
        if (matrix->kind == INKLING_MATRIX_BF16) {
            uint16_t *bf16 = (uint16_t *)context->weights;
            for (size_t i = 0; i < data_bytes / 2; i++)
                bf16[i] = (uint16_t)((uint16_t)context->weights[2 * i] |
                                     ((uint16_t)context->weights[2 * i + 1] << 8));
            if (!inkling_bf16_matvec(bf16, data_bytes / 2, count, matrix->columns,
                    input, matrix->columns, output + row, count)) return 0;
        } else {
            size_t scales_count = count * scale_bytes;
            if (cached_data == NULL) {
                if (!inkling_size_mul(source_row, scale_bytes, &offset) ||
                    !inkling_context_read(context, matrix->scales, offset,
                        scales_count, context->weights + data_bytes)) return 0;
                scales = context->weights + data_bytes;
            }
            uint64_t shape[3] = {1, count, matrix->columns};
            uint64_t packed_shape[3] = {1, count, matrix->columns / 2};
            uint64_t scale_shape[3] = {1, count, matrix->columns / 16};
            InklingNvfp4Layout layout;
            if (!inkling_nvfp4_layout(shape, packed_shape, scale_shape, 1, &layout) ||
                !inkling_nvfp4_matvec(&layout, 0, data, data_bytes, scales, scales_count,
                    matrix->scale2 + expert, 1, input, matrix->columns, output + row, count)) return 0;
        }
        row += count;
    }
    return 1;
}
