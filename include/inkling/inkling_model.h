#ifndef INKLING_MODEL_H
#define INKLING_MODEL_H
#include "inkling.h"
typedef struct InklingModel InklingModel;
typedef struct InklingContext InklingContext;
/* Constructors clear *out and publish an owned object only on success. */
int inkling_model_open(const char *directory, InklingModel **out);
/* Free all contexts before closing their model. */
void inkling_model_close(InklingModel *model);
size_t inkling_model_logits_width(const InklingModel *model);
int inkling_context_create(const InklingModel *model, size_t max_tokens, InklingContext **out);
void inkling_context_reset(InklingContext *context);
void inkling_context_free(InklingContext *context);
/* Every forward starts a fresh sequence. Discard logits after any failure. */
int inkling_forward(InklingContext *context, const uint32_t *ids, size_t tokens,
                    float *logits, size_t logits_count);

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
