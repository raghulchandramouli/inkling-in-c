#ifndef INKLING_MODEL_H
#define INKLING_MODEL_H
#include "inkling.h"
typedef struct InklingModel InklingModel;
typedef struct InklingContext InklingContext;
/* Constructors clear *out and publish an owned object only on success. */
typedef struct { const char *packed_trunk_path; } InklingModelOptions;
#define INKLING_MODEL_OPTIONS_INIT {NULL}
int inkling_model_open(const char *directory, InklingModel **out);
int inkling_model_open_with_options(const char *directory, const InklingModelOptions *options,
                                     InklingModel **out);
/* Free all contexts before closing their model. */
void inkling_model_close(InklingModel *model);
size_t inkling_model_logits_width(const InklingModel *model);
typedef struct {
    size_t max_tokens;
    size_t expert_cache_bytes;
    size_t weight_buffer_bytes;
    int incremental;
} InklingContextOptions;
#define INKLING_CONTEXT_OPTIONS_INIT {0, 0, 1048576, 1}
typedef struct {
    uint64_t bytes_read, read_calls, cache_hits, cache_misses, cache_evictions, tokens_processed;
    size_t cache_bytes, cache_peak_bytes, cache_capacity;
} InklingRuntimeStats;
typedef enum {
    INKLING_CACHE_HIT, INKLING_CACHE_MISS, INKLING_CACHE_BYPASS, INKLING_CACHE_EVICT
} InklingCacheEventKind;
typedef struct {
    InklingCacheEventKind kind;
    size_t layer, expert, projection, bytes, resident_bytes;
} InklingCacheEvent;
typedef void (*InklingCacheTrace)(void *user, const InklingCacheEvent *event);
void inkling_context_cache_trace(InklingContext *context, InklingCacheTrace trace, void *user);
int inkling_context_create(const InklingModel *model, size_t max_tokens, InklingContext **out);
int inkling_context_create_with_options(const InklingModel *model,
                                        const InklingContextOptions *options, InklingContext **out);
void inkling_context_get_stats(const InklingContext *context, InklingRuntimeStats *stats);
void inkling_context_reset_stats(InklingContext *context);
void inkling_context_reset(InklingContext *context);
void inkling_context_free(InklingContext *context);
/* Every forward starts a fresh sequence. Discard logits after any failure. */
int inkling_forward(InklingContext *context, const uint32_t *ids, size_t tokens,
                    float *logits, size_t logits_count);

/* Prefill resets sequence state and returns the last prompt token's logits.
 * Decode appends one token. Failures invalidate sequence state; prefill or reset
 * before retrying. Neither path changes caller logits on validation failure. */
int inkling_prefill(InklingContext *context, const uint32_t *ids, size_t tokens,
                     float *logits, size_t logits_count);
int inkling_decode(InklingContext *context, uint32_t id, float *logits, size_t logits_count);
size_t inkling_context_position(const InklingContext *context);
size_t inkling_model_max_length(const InklingModel *model);
typedef struct {
    size_t max_new_tokens;
    double temperature;
    size_t top_k;
    double top_p;
    uint64_t seed;
} InklingGenerateOptions;
#define INKLING_GENERATE_OPTIONS_INIT {0, 0.0, 0, 1.0, UINT64_C(0)}
typedef enum {
    INKLING_STOP_LENGTH = 1,
    INKLING_STOP_CONTEXT,
    INKLING_STOP_EOS
} InklingStopReason;
typedef struct {
    size_t generated_count;
    InklingStopReason stop_reason;
    InklingGenerateOptions effective;
} InklingGenerateResult;
/* Each call replays from seed. Output IDs and result are unchanged on failure.
 * Capacity must cover min(max_new_tokens, usable capacity minus prompt count).
 * Generated EOS is included. Stop priority is EOS, requested length, context.
 * Inputs, output's declared range and result must not overlap writable outputs. */
int inkling_generate(InklingContext *context, const uint32_t *prompt, size_t prompt_count,
                     const InklingGenerateOptions *options, uint32_t *output,
                     size_t output_capacity, InklingGenerateResult *result);
#endif
