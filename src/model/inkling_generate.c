#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling_alloc.h"
#include "inkling_model_internal.h"

struct InklingGeneration {
    uint32_t *prefix;
    float *logits;
    InklingCandidate *candidates;
    uint64_t state;
};

static int failure(const char *message)
{
    fprintf(stderr, "inkling generate: %s\n", message);
    return 0;
}

void inkling_generation_free(struct InklingGeneration *generation)
{
    if (generation == NULL) return;
    free(generation->prefix);
    free(generation->logits);
    free(generation->candidates);
    free(generation);
}

int inkling_generate_options(const InklingGenerateOptions *options, size_t width,
                             InklingGenerateOptions *effective)
{
    if (options == NULL || effective == NULL || width == 0 || width > UINT32_MAX ||
        !isfinite(options->temperature) || options->temperature < 0 ||
        !isfinite(options->top_p) || options->top_p <= 0 || options->top_p > 1) return 0;
    *effective = *options;
    if (options->temperature == 0) {
        effective->temperature = 0;
        effective->top_k = 1;
        effective->top_p = 1;
    } else if (options->top_k == 0 || options->top_k > width) effective->top_k = width;
    return 1;
}

static int rank_candidate(const void *left, const void *right)
{
    const InklingCandidate *a = left, *b = right;
    if (a->logit > b->logit) return -1;
    if (a->logit < b->logit) return 1;
    return (a->id > b->id) - (a->id < b->id);
}

int inkling_sample(const float *logits, size_t width, const InklingGenerateOptions *options,
                    InklingCandidate *candidates, uint64_t *state, uint32_t *id)
{
    if (logits == NULL || options == NULL || state == NULL || id == NULL ||
        width == 0 || width > UINT32_MAX || !isfinite(options->temperature) ||
        options->temperature < 0 || !isfinite(options->top_p) ||
        options->top_p <= 0 || options->top_p > 1 ||
        options->top_k == 0 || options->top_k > width) return 0;
    size_t best = 0;
    for (size_t i = 0; i < width; i++) {
        if (!isfinite(logits[i])) return 0;
        if (logits[i] > logits[best]) best = i;
    }
    if (options->temperature == 0) { *id = (uint32_t)best; return 1; }
    if (candidates == NULL) return 0;
    for (size_t i = 0; i < width; i++) {
        candidates[i].logit = logits[i];
        candidates[i].id = (uint32_t)i;
    }
    qsort(candidates, width, sizeof(*candidates), rank_candidate);
    double total = 0;
    for (size_t i = 0; i < options->top_k; i++) {
        double gap = (double)candidates[0].logit - (double)candidates[i].logit;
        double mass = gap == 0 ? 1 :
            options->temperature < 1 && gap > DBL_MAX * options->temperature ? 0 :
            exp(-gap / options->temperature);
        candidates[i].mass = mass;
        total += mass;
    }
    size_t retained = options->top_k;
    if (options->top_p < 1) {
        double cumulative = 0, threshold = options->top_p * total;
        for (size_t i = 0; i < retained; i++) {
            cumulative += candidates[i].mass;
            if (cumulative >= threshold) { retained = i + 1; total = cumulative; break; }
        }
    }
    *state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t word = *state;
    word = (word ^ (word >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    word = (word ^ (word >> 27)) * UINT64_C(0x94d049bb133111eb);
    word ^= word >> 31;
    double target = (double)(word >> 11) * 0x1p-53 * total, cumulative = 0;
    uint32_t last = candidates[0].id;
    for (size_t i = 0; i < retained; i++) {
        if (candidates[i].mass == 0) continue;
        last = candidates[i].id;
        cumulative += candidates[i].mass;
        if (cumulative > target) { *id = last; return 1; }
    }
    *id = last;
    return 1;
}

static int valid_range(const void *pointer, size_t count, size_t element, size_t *bytes)
{
    return inkling_size_mul(count, element, bytes) && *bytes <= (size_t)PTRDIFF_MAX &&
        (*bytes == 0 || (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - *bytes));
}

static int overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    if (a_bytes == 0 || b_bytes == 0) return 0;
    uintptr_t left = (uintptr_t)a, right = (uintptr_t)b;
    return left <= right ? right - left < a_bytes : left - right < b_bytes;
}

static int prepare(InklingContext *context, size_t capacity, size_t width)
{
    if (context->generation != NULL) return 1;
    void *allocation = NULL;
    if (!inkling_alloc_array(1, sizeof(struct InklingGeneration), &allocation)) return 0;
    struct InklingGeneration *generation = allocation;
    memset(generation, 0, sizeof(*generation));
    if (!inkling_alloc_array(capacity, sizeof(uint32_t), &allocation)) goto fail;
    generation->prefix = allocation;
    if (!inkling_alloc_array(width, sizeof(float), &allocation)) goto fail;
    generation->logits = allocation;
    if (!inkling_alloc_array(width, sizeof(InklingCandidate), &allocation)) goto fail;
    generation->candidates = allocation;
    context->generation = generation;
    return 1;
fail:
    inkling_generation_free(generation);
    return 0;
}

int inkling_generate(InklingContext *context, const uint32_t *prompt, size_t prompt_count,
                     const InklingGenerateOptions *options, uint32_t *output,
                     size_t output_capacity, InklingGenerateResult *result)
{
    size_t prompt_bytes, output_bytes, options_bytes, result_bytes, total;
    if (context == NULL || context->model == NULL || !context->model->complete ||
        prompt_count == 0 ||
        !valid_range(prompt, prompt_count, sizeof(*prompt), &prompt_bytes) ||
        !valid_range(output, output_capacity, sizeof(*output), &output_bytes) ||
        !valid_range(options, 1, sizeof(*options), &options_bytes) ||
        !valid_range(result, 1, sizeof(*result), &result_bytes) ||
        overlaps(output, output_bytes, prompt, prompt_bytes) ||
        overlaps(output, output_bytes, options, options_bytes) ||
        overlaps(output, output_bytes, result, result_bytes) ||
        overlaps(result, result_bytes, prompt, prompt_bytes) ||
        overlaps(result, result_bytes, options, options_bytes))
        return failure("invalid context, buffers, or overlapping ranges");
    InklingGenerateResult staged;
    memset(&staged, 0, sizeof(staged));
    size_t width = inkling_model_logits_width(context->model);
    if (!inkling_generate_options(options, width, &staged.effective) ||
        !inkling_size_add(prompt_count, options->max_new_tokens, &total))
        return failure("invalid sampler options or prompt plus request overflow");
    size_t usable = inkling_model_max_length(context->model);
    if (context->capacity < usable) usable = context->capacity;
    if (prompt_count > usable) return failure("prompt exceeds context capacity");
    size_t required = usable - prompt_count;
    if (required > options->max_new_tokens) required = options->max_new_tokens;
    if (output_capacity < required) return failure("insufficient output capacity");
    for (size_t i = 0; i < prompt_count; i++)
        if (prompt[i] >= context->model->config.vocab_size) return failure("prompt token outside vocabulary");
    staged.stop_reason = options->max_new_tokens == 0 ? INKLING_STOP_LENGTH : INKLING_STOP_CONTEXT;
    if (required != 0) {
        if (!prepare(context, usable, width)) return failure("generation allocation failed");
        struct InklingGeneration *generation = context->generation;
        memcpy(generation->prefix, prompt, prompt_bytes);
        generation->state = options->seed;
        while (staged.generated_count < required) {
            size_t length = prompt_count + staged.generated_count;
            uint32_t token;
            if (!inkling_forward_last(context, generation->prefix, length, generation->logits, width)) return 0;
            if (!inkling_sample(generation->logits, width, &staged.effective,
                               generation->candidates, &generation->state, &token))
                return failure("sampling failed");
            generation->prefix[length] = token;
            staged.generated_count++;
            if (token == context->model->config.eos_token_id) {
                staged.stop_reason = INKLING_STOP_EOS;
                break;
            }
            if (staged.generated_count == options->max_new_tokens) {
                staged.stop_reason = INKLING_STOP_LENGTH;
                break;
            }
        }
        memcpy(output, generation->prefix + prompt_count, staged.generated_count * sizeof(*output));
    }
    *result = staged;
    return 1;
}
