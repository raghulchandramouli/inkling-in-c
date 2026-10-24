#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling_alloc.h"
#include "inkling_bf16.h"
#include "inkling_math.h"
#include "inkling_model_internal.h"

static int failure(const char *message)
{
    fprintf(stderr, "inkling forward: %s\n", message);
    return 0;
}

static int allocate_floats(float **out, size_t count)
{
    void *allocation = NULL;
    if (!inkling_alloc_array(count, sizeof(float), &allocation)) return 0;
    *out = allocation;
    return 1;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    uintptr_t left = (uintptr_t)a, right = (uintptr_t)b;
    return left <= right ? right - left < a_bytes : left - right < b_bytes;
}

static int finite_values(const float *values, size_t count)
{
    for (size_t i = 0; i < count; i++) if (!isfinite(values[i])) return 0;
    return 1;
}

static void trace_values(InklingContext *context, const char *name,
                         const float *values, size_t count)
{
    if (context->trace != NULL)
        context->trace(context->trace_user, name, values, NULL, count);
}

void inkling_context_trace(InklingContext *context, InklingTrace trace, void *user)
{
    if (context == NULL) return;
    context->trace = trace;
    context->trace_user = user;
}

typedef struct {
    size_t capacity;
    float *keys, *values;
    float *key_history, *value_history, *attn_history, *mlp_history;
} InklingLayerState;

struct InklingSequence {
    InklingLayerState *layers;
    size_t layer_count, position;
    int valid;
};

static void sequence_free(struct InklingSequence *sequence)
{
    if (sequence == NULL) return;
    for (size_t i = 0; i < sequence->layer_count; i++) {
        InklingLayerState *state = &sequence->layers[i];
        free(state->keys); free(state->values);
        free(state->key_history); free(state->value_history);
        free(state->attn_history); free(state->mlp_history);
    }
    free(sequence->layers);
    free(sequence);
}

void inkling_context_reset(InklingContext *context)
{
    if (context == NULL) return;
    if (context->history != NULL)
        memset(context->history, 0, context->history_count * sizeof(float));
    struct InklingSequence *sequence = context->sequence;
    if (sequence == NULL) return;
    sequence->position = 0;
    sequence->valid = 1;
    for (size_t i = 0; i < sequence->layer_count; i++) {
        InklingLayerState *state = &sequence->layers[i];
        size_t kv = context->model->layers[i].k.rows;
        size_t hidden = context->model->config.hidden_size;
        memset(state->key_history, 0, 3 * kv * sizeof(float));
        memset(state->value_history, 0, 3 * kv * sizeof(float));
        memset(state->attn_history, 0, 3 * hidden * sizeof(float));
        memset(state->mlp_history, 0, 3 * hidden * sizeof(float));
    }
}

void inkling_context_free(InklingContext *context)
{
    if (context == NULL) return;
    free(context->hidden[0]); free(context->hidden[1]);
    free(context->norm); free(context->block);
    free(context->q); free(context->k); free(context->v);
    free(context->relative); free(context->heads);
    free(context->scores); free(context->bias); free(context->history);
    free(context->pair); free(context->activation); free(context->expert);
    free(context->routed); free(context->shared);
    free(context->router); free(context->mixture);
    free(context->selected); free(context->weights);
    inkling_generation_free(context->generation);
    sequence_free(context->sequence);
    inkling_expert_cache_free(context->expert_cache);
    free(context);
}

static int scratch_reserve(InklingContext *context, size_t tokens)
{
    if (tokens <= context->scratch_capacity) return 1;
    size_t q_width = 0, kv_width = 0, relative_width = 0;
    for (size_t i = 0; i < context->model->layer_count; i++) {
        const InklingLayer *layer = &context->model->layers[i];
        if (layer->q.rows > q_width) q_width = layer->q.rows;
        if (layer->k.rows > kv_width) kv_width = layer->k.rows;
        if (layer->r.rows > relative_width) relative_width = layer->r.rows;
    }
    size_t hidden_count, q_count, kv_count, relative_count;
    if (!inkling_size_mul(tokens, context->model->config.hidden_size, &hidden_count) ||
        !inkling_size_mul(tokens, q_width, &q_count) ||
        !inkling_size_mul(tokens, kv_width, &kv_count) ||
        !inkling_size_mul(tokens, relative_width, &relative_count)) return 0;
    float *buffers[9] = {NULL};
    const size_t counts[9] = {hidden_count, hidden_count, hidden_count, hidden_count,
                             q_count, kv_count, kv_count, relative_count, q_count};
    for (size_t i = 0; i < 9; i++) {
        if (!allocate_floats(&buffers[i], counts[i])) {
            for (size_t j = 0; j < 9; j++) free(buffers[j]);
            return 0;
        }
    }
    float **destinations[9] = {&context->hidden[0], &context->hidden[1],
        &context->norm, &context->block, &context->q, &context->k, &context->v,
        &context->relative, &context->heads};
    for (size_t i = 0; i < 9; i++) {
        free(*destinations[i]);
        *destinations[i] = buffers[i];
    }
    context->scratch_capacity = tokens;
    return 1;
}

static int sequence_create(InklingContext *context)
{
    struct InklingSequence *sequence = calloc(1, sizeof(*sequence));
    if (sequence == NULL) return 0;
    context->sequence = sequence;
    if (!inkling_alloc_array(context->model->layer_count, sizeof(*sequence->layers),
                             (void **)&sequence->layers)) return 0;
    memset(sequence->layers, 0, context->model->layer_count * sizeof(*sequence->layers));
    sequence->layer_count = context->model->layer_count;
    for (size_t i = 0; i < sequence->layer_count; i++) {
        const InklingLayer *layer = &context->model->layers[i];
        InklingLayerState *state = &sequence->layers[i];
        state->capacity = layer->window != 0 && layer->window < context->capacity
            ? layer->window : context->capacity;
        size_t count, kv_history, hidden_history;
        if (!inkling_size_mul(state->capacity, layer->k.rows, &count) ||
            !inkling_size_mul(layer->k.rows, 3, &kv_history) ||
            !inkling_size_mul(context->model->config.hidden_size, 3, &hidden_history) ||
            !allocate_floats(&state->keys, count) ||
            !allocate_floats(&state->values, count) ||
            !allocate_floats(&state->key_history, kv_history) ||
            !allocate_floats(&state->value_history, kv_history) ||
            !allocate_floats(&state->attn_history, hidden_history) ||
            !allocate_floats(&state->mlp_history, hidden_history)) return 0;
    }
    return 1;
}

static int matrix_row_fits(const InklingMatrix *matrix, size_t bytes)
{
    size_t row_bytes;
    if (matrix->rows == 0) return 1;
    if (matrix->kind == INKLING_MATRIX_BF16)
        return inkling_size_mul(matrix->columns, 2, &row_bytes) && row_bytes <= bytes;
    return inkling_size_add(matrix->columns / 2, matrix->columns / 16, &row_bytes) &&
        row_bytes <= bytes;
}

int inkling_context_create_with_options(const InklingModel *model,
    const InklingContextOptions *options, InklingContext **out)
{
    if (out == NULL) return failure("missing context output");
    *out = NULL;
    if (model == NULL || options == NULL || options->max_tokens == 0 ||
        options->max_tokens > model->config.model_max_length || model->layer_count == 0 ||
        model->layers == NULL || !inkling_config_is_valid(&model->config) ||
        !model->config.execution_fields_valid || options->weight_buffer_bytes == 0 ||
        options->weight_buffer_bytes > (size_t)PTRDIFF_MAX ||
        options->expert_cache_bytes > (size_t)PTRDIFF_MAX ||
        (options->incremental != 0 && options->incremental != 1))
        return failure("invalid model or context options");
    if (!matrix_row_fits(&model->embed, options->weight_buffer_bytes) ||
        !matrix_row_fits(&model->unembed, options->weight_buffer_bytes))
        return failure("weight buffer cannot fit embedding or output matrix row");
    size_t hidden = model->config.hidden_size;
    size_t kv_width = 0, pair_width = 0;
    for (size_t i = 0; i < model->layer_count; i++) {
        const InklingLayer *layer = &model->layers[i];
        const InklingMatrix *matrices[] = {&layer->q, &layer->k, &layer->v, &layer->r,
            &layer->o, &layer->w13, &layer->down, &layer->shared_w13,
            &layer->shared_down, &layer->router};
        for (size_t j = 0; j < sizeof(matrices) / sizeof(matrices[0]); j++)
            if (!matrix_row_fits(matrices[j], options->weight_buffer_bytes))
                return failure("weight buffer cannot fit a matrix row");
        if (layer->k.rows > kv_width) kv_width = layer->k.rows;
        if (layer->w13.rows > pair_width) pair_width = layer->w13.rows;
        if (layer->shared_w13.rows > pair_width) pair_width = layer->shared_w13.rows;
    }
    size_t history_count, router_count, mixture_count, embedding_bytes;
    if (!inkling_size_mul(hidden > kv_width ? hidden : kv_width, 3, &history_count) ||
        !inkling_size_mul(hidden, 2, &embedding_bytes) ||
        !inkling_size_add(model->config.num_routed_experts,
                          model->config.num_shared_experts, &router_count) ||
        !inkling_size_add(model->config.num_experts_per_token,
                          model->config.num_shared_experts, &mixture_count) ||
        options->weight_buffer_bytes < embedding_bytes)
        return failure("context dimensions overflow or weight buffer too small");
    InklingContext *context = calloc(1, sizeof(*context));
    if (context == NULL) return failure("cannot allocate context");
    context->model = model;
    context->capacity = options->max_tokens;
    context->history_count = history_count;
    context->weight_bytes = options->weight_buffer_bytes;
    context->expert_cache_capacity = options->expert_cache_bytes;
    context->stats.cache_capacity = options->expert_cache_bytes;
    context->incremental = options->incremental;
    void *selected = NULL, *weights = NULL;
    if (!scratch_reserve(context, options->incremental ? 1 : options->max_tokens) ||
        !allocate_floats(&context->scores, options->max_tokens) ||
        !allocate_floats(&context->bias, options->max_tokens) ||
        !allocate_floats(&context->history, history_count) ||
        !allocate_floats(&context->pair, pair_width) ||
        !allocate_floats(&context->activation, pair_width / 2) ||
        !allocate_floats(&context->expert, hidden) ||
        !allocate_floats(&context->routed, hidden) ||
        !allocate_floats(&context->shared, hidden) ||
        !allocate_floats(&context->router, router_count) ||
        !allocate_floats(&context->mixture, mixture_count) ||
        !inkling_alloc_array(model->config.num_experts_per_token, sizeof(size_t), &selected) ||
        !inkling_alloc_array(context->weight_bytes, 1, &weights) ||
        (context->incremental && !sequence_create(context))) {
        free(selected); free(weights); inkling_context_free(context);
        return failure("cannot allocate context buffers");
    }
    context->selected = selected;
    context->weights = weights;
    inkling_context_reset(context);
    inkling_context_reset_stats(context);
    *out = context;
    return 1;
}

int inkling_context_create(const InklingModel *model, size_t max_tokens,
                           InklingContext **out)
{
    InklingContextOptions options = INKLING_CONTEXT_OPTIONS_INIT;
    options.max_tokens = max_tokens;
    return inkling_context_create_with_options(model, &options, out);
}

static int block_args(InklingContext *context, const InklingLayer *layer,
                      const float *input, size_t tokens, float *output)
{
    if (context == NULL || layer == NULL || input == NULL || output == NULL ||
        tokens == 0 || tokens > context->capacity) return 0;
    int found = 0;
    for (size_t i = 0; i < context->model->layer_count; i++)
        if (layer == &context->model->layers[i]) found = 1;
    size_t count, bytes;
    if (!inkling_size_mul(tokens, context->model->config.hidden_size, &count) ||
        !inkling_size_mul(count, sizeof(float), &bytes) || bytes > (size_t)PTRDIFF_MAX) return 0;
    return found && !overlap(input, bytes, output, bytes) &&
        finite_values(input, bytes / sizeof(float)) && scratch_reserve(context, tokens);
}

static int convolve(InklingContext *context, float *values, size_t tokens,
                    size_t channels, const float *weights)
{
    memset(context->history, 0, context->history_count * sizeof(float));
    return inkling_conv4(values, tokens * channels, tokens, channels,
                         weights, channels * 4, context->history,
                         context->history_count, values, tokens * channels);
}

int inkling_attention_forward(InklingContext *context, const InklingLayer *layer,
                              const float *input, size_t tokens,
                              size_t position_start, float *output)
{
    if (!block_args(context, layer, input, tokens, output) ||
        position_start > SIZE_MAX - tokens)
        return failure("invalid attention buffers, layer, capacity or positions");
    const InklingConfig *config = &context->model->config;
    size_t hidden = config->hidden_size, q_width = layer->q.rows;
    size_t kv_width = layer->k.rows, relative_width = layer->r.rows;
    size_t dimension = layer->head_dim, relative = config->relative_dimension;
    for (size_t t = 0; t < tokens; t++) {
        const float *x = input + t * hidden;
        if (!inkling_project(context, &layer->q, 0, q_width, x, context->q + t * q_width) ||
            !inkling_project(context, &layer->k, 0, kv_width, x, context->k + t * kv_width) ||
            !inkling_project(context, &layer->v, 0, kv_width, x, context->v + t * kv_width) ||
            !inkling_project(context, &layer->r, 0, relative_width, x, context->relative + t * relative_width))
            return failure("attention projection failed");
    }
    trace_values(context, "q_proj", context->q, tokens * q_width);
    trace_values(context, "k_proj", context->k, tokens * kv_width);
    trace_values(context, "v_proj", context->v, tokens * kv_width);
    trace_values(context, "r_proj", context->relative, tokens * relative_width);
    if (!convolve(context, context->k, tokens, kv_width, layer->k_conv) ||
        !convolve(context, context->v, tokens, kv_width, layer->v_conv))
        return failure("key/value convolution failed");
    trace_values(context, "k_conv", context->k, tokens * kv_width);
    trace_values(context, "v_conv", context->v, tokens * kv_width);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < layer->heads; h++) {
            float *q = context->q + t * q_width + h * dimension;
            if (!inkling_rmsnorm(q, dimension, layer->q_norm, dimension,
                    config->rms_norm_epsilon, q, dimension)) return failure("query norm failed");
        }
        for (size_t h = 0; h < layer->kv_heads; h++) {
            float *k = context->k + t * kv_width + h * dimension;
            if (!inkling_rmsnorm(k, dimension, layer->k_norm, dimension,
                    config->rms_norm_epsilon, k, dimension)) return failure("key norm failed");
        }
    }
    trace_values(context, "q_norm", context->q, tokens * q_width);
    trace_values(context, "k_norm", context->k, tokens * kv_width);
    for (size_t t = 0; t < tokens; t++) {
        float tau = 1.0f;
        if (layer->window == 0) {
            float ratio = (float)(position_start + t + 1) / (float)config->log_scaling_n_floor;
            tau += config->log_scaling_alpha * logf(fmaxf(1.0f, ratio));
        }
        for (size_t j = 0; j < q_width; j++) context->q[t * q_width + j] *= tau;
    }
    if (!finite_values(context->q, tokens * q_width)) return failure("query scaling overflow");
    trace_values(context, "query_scaled", context->q, tokens * q_width);
    for (size_t t = 0; t < tokens; t++) {
        float tau = 1.0f;
        if (layer->window == 0) {
            float ratio = (float)(position_start + t + 1) / (float)config->log_scaling_n_floor;
            tau += config->log_scaling_alpha * logf(fmaxf(1.0f, ratio));
        }
        if (!isfinite(tau)) return failure("nonfinite attention scaling");
        for (size_t h = 0; h < layer->heads; h++) {
            const float *q = context->q + t * q_width + h * dimension;
            const float *r = context->relative + t * relative_width + h * relative;
            size_t kv = h / (layer->heads / layer->kv_heads);
            for (size_t s = 0; s < tokens; s++) {
                float bias = 0;
                if (s <= t && t - s < layer->extent)
                    for (size_t j = 0; j < relative; j++)
                        bias += r[j] * layer->relative[j * layer->extent + t - s];
                context->bias[s] = bias;
                if (s > t || (layer->window != 0 && t - s >= layer->window)) {
                    context->scores[s] = -INFINITY;
                    continue;
                }
                const float *k = context->k + s * kv_width + kv * dimension;
                float score = 0;
                for (size_t j = 0; j < dimension; j++) score += q[j] * k[j];
                context->scores[s] = score / (float)dimension + context->bias[s] * tau;
            }
            if (!finite_values(context->bias, tokens)) return failure("relative bias overflow");
            trace_values(context, "relative_bias", context->bias, tokens);
            for (size_t s = 0; s < tokens; s++) context->bias[s] *= tau;
            if (!finite_values(context->bias, tokens)) return failure("relative scaling overflow");
            trace_values(context, "relative_scaled", context->bias, tokens);
            trace_values(context, "attention_scores", context->scores, tokens);
            if (!inkling_softmax(context->scores, tokens, context->scores, tokens))
                return failure("attention softmax failed");
            trace_values(context, "attention_probs", context->scores, tokens);
            float *head = context->heads + t * q_width + h * dimension;
            for (size_t j = 0; j < dimension; j++) {
                float value = 0;
                for (size_t s = 0; s <= t; s++)
                    value += context->scores[s] * context->v[s * kv_width + kv * dimension + j];
                head[j] = value;
            }
        }
    }
    trace_values(context, "attention_heads", context->heads, tokens * q_width);
    for (size_t t = 0; t < tokens; t++)
        if (!inkling_project(context, &layer->o, 0, hidden, context->heads + t * q_width,
                      output + t * hidden)) return failure("attention output projection failed");
    trace_values(context, "attention_output", output, tokens * hidden);
    return 1;
}

static int expert_forward(InklingContext *context, const InklingMatrix *w13,
                          const InklingMatrix *down, size_t expert,
                          const float *input, float gamma, int shared)
{
    size_t width = w13->rows / 2;
    if (!inkling_project(context, w13, expert, w13->rows, input, context->pair)) return 0;
    for (size_t j = 0; j < width; j++) context->activation[j] = context->pair[2 * j];
    if (!inkling_silu(context->activation, width, context->activation, width)) return 0;
    for (size_t j = 0; j < width; j++) {
        context->activation[j] *= context->pair[2 * j + 1];
        if (shared) context->activation[j] *= gamma;
    }
    return inkling_project(context, down, expert, down->rows, context->activation, context->expert);
}

int inkling_router_forward(const InklingModel *model, const InklingLayer *layer,
                           const float *logits, size_t *selected, float *weights)
{
    if (model == NULL || layer == NULL || logits == NULL || selected == NULL ||
        weights == NULL || layer->correction == NULL)
        return failure("missing router buffers");
    const InklingConfig *config = &model->config;
    size_t routed = config->num_routed_experts, shared = config->num_shared_experts;
    size_t count = config->num_experts_per_token, logits_count, mixture_count;
    size_t logits_bytes, mixture_bytes, selected_bytes;
    if (routed == 0 || count == 0 || count > routed ||
        !inkling_size_add(routed, shared, &logits_count) ||
        !inkling_size_add(count, shared, &mixture_count) ||
        !inkling_size_mul(logits_count, sizeof(float), &logits_bytes) ||
        !inkling_size_mul(mixture_count, sizeof(float), &mixture_bytes) ||
        !inkling_size_mul(count, sizeof(size_t), &selected_bytes) ||
        logits_bytes > (size_t)PTRDIFF_MAX || mixture_bytes > (size_t)PTRDIFF_MAX ||
        selected_bytes > (size_t)PTRDIFF_MAX ||
        overlap(logits, logits_bytes, weights, mixture_bytes) ||
        overlap(logits, logits_bytes, selected, selected_bytes) ||
        overlap(weights, mixture_bytes, selected, selected_bytes) ||
        overlap(layer->correction, routed * sizeof(float), weights, mixture_bytes) ||
        overlap(layer->correction, routed * sizeof(float), selected, selected_bytes) ||
        !finite_values(logits, logits_count) ||
        !finite_values(layer->correction, routed) ||
        !isfinite(config->route_scale) || !isfinite(layer->global_scale))
        return failure("invalid router dimensions, buffers or weights");
    float *choice = NULL;
    if (!allocate_floats(&choice, routed)) return failure("cannot allocate router scores");
    int ok = inkling_sigmoid(logits, routed, choice, routed);
    for (size_t e = 0; e < routed; e++) choice[e] += layer->correction[e];
    ok = ok && finite_values(choice, routed) &&
        inkling_topk(choice, routed, count, selected, count);
    free(choice);
    if (!ok) return failure("router selection failed");
    for (size_t i = 0; i < mixture_count; i++) {
        float z = logits[i < count ? selected[i] : routed + i - count];
        weights[i] = z >= 0 ? -log1pf(expf(-z)) : z - log1pf(expf(z));
    }
    float maximum = weights[0];
    for (size_t i = 1; i < mixture_count; i++)
        if (weights[i] > maximum) maximum = weights[i];
    double sum = 0;
    for (size_t i = 0; i < mixture_count; i++) sum += exp((double)weights[i] - maximum);
    float normalizer = maximum + (float)log(sum);
    for (size_t i = 0; i < mixture_count; i++)
        weights[i] = (expf(weights[i] - normalizer) * config->route_scale) * layer->global_scale;
    if (!finite_values(weights, mixture_count)) return failure("router weights overflow");
    return 1;
}

int inkling_mlp_forward(InklingContext *context, const InklingLayer *layer,
                        const float *input, size_t tokens, float *output)
{
    if (!block_args(context, layer, input, tokens, output))
        return failure("invalid MLP buffers, layer or capacity");
    const InklingConfig *config = &context->model->config;
    size_t hidden = config->hidden_size, routed = config->num_routed_experts;
    size_t shared = config->num_shared_experts, selected = config->num_experts_per_token;
    for (size_t t = 0; t < tokens; t++) {
        const float *x = input + t * hidden;
        float *y = output + t * hidden;
        if (layer->dense) {
            if (!expert_forward(context, &layer->w13, &layer->down, 0, x, 1.0f, 0))
                return failure("dense MLP projection failed");
            for (size_t j = 0; j < hidden; j++) y[j] = context->expert[j] * layer->global_scale;
        } else {
            if (!inkling_project(context, &layer->router, 0, routed + shared, x, context->router) ||
                !inkling_router_forward(context->model, layer, context->router,
                                         context->selected, context->mixture))
                return failure("router projection or selection failed");
            trace_values(context, "router_logits", context->router, routed + shared);
            if (context->trace != NULL)
                context->trace(context->trace_user, "router_selected", NULL, context->selected, selected);
            trace_values(context, "router_weights", context->mixture, selected);
            trace_values(context, "shared_gammas", context->mixture + selected, shared);
            for (size_t i = 1; i < selected; i++) {
                size_t id = context->selected[i], j = i;
                float weight = context->mixture[i];
                while (j != 0 && context->selected[j - 1] > id) {
                    context->selected[j] = context->selected[j - 1];
                    context->mixture[j] = context->mixture[j - 1];
                    j--;
                }
                context->selected[j] = id;
                context->mixture[j] = weight;
            }
            memset(context->routed, 0, hidden * sizeof(float));
            memset(context->shared, 0, hidden * sizeof(float));
            for (size_t i = 0; i < selected; i++) {
                if (!expert_forward(context, &layer->w13, &layer->down,
                                    context->selected[i], x, 1.0f, 0))
                    return failure("routed expert projection failed");
                for (size_t j = 0; j < hidden; j++)
                    context->routed[j] += context->expert[j] * context->mixture[i];
            }
            for (size_t i = 0; i < shared; i++) {
                if (!expert_forward(context, &layer->shared_w13, &layer->shared_down,
                                    i, x, context->mixture[selected + i], 1))
                    return failure("shared expert projection failed");
                for (size_t j = 0; j < hidden; j++) context->shared[j] += context->expert[j];
            }
            trace_values(context, "routed_output", context->routed, hidden);
            trace_values(context, "shared_output", context->shared, hidden);
            for (size_t j = 0; j < hidden; j++) y[j] = context->routed[j] + context->shared[j];
        }
        if (!finite_values(y, hidden)) return failure("MLP output overflow");
    }
    trace_values(context, "mlp_output", output, tokens * hidden);
    return 1;
}

int inkling_layer_forward(InklingContext *context, const InklingLayer *layer,
                          const float *input, size_t tokens, float *output)
{
    if (!block_args(context, layer, input, tokens, output))
        return failure("invalid layer buffers or capacity");
    const InklingConfig *config = &context->model->config;
    size_t hidden = config->hidden_size, count = tokens * hidden;
    for (size_t t = 0; t < tokens; t++)
        if (!inkling_rmsnorm(input + t * hidden, hidden, layer->attn_norm, hidden,
                config->rms_norm_epsilon, context->norm + t * hidden, hidden))
            return failure("attention input norm failed");
    trace_values(context, "attn_norm", context->norm, count);
    if (!inkling_attention_forward(context, layer, context->norm, tokens, 0, context->block)) return 0;
    if (!convolve(context, context->block, tokens, hidden, layer->attn_conv))
        return failure("attention output convolution failed");
    trace_values(context, "attn_conv", context->block, count);
    for (size_t i = 0; i < count; i++) output[i] = input[i] + context->block[i];
    if (!finite_values(output, count)) return failure("attention residual overflow");
    trace_values(context, "post_attention", output, count);
    for (size_t t = 0; t < tokens; t++)
        if (!inkling_rmsnorm(output + t * hidden, hidden, layer->mlp_norm, hidden,
                config->rms_norm_epsilon, context->norm + t * hidden, hidden))
            return failure("MLP input norm failed");
    trace_values(context, "mlp_norm", context->norm, count);
    if (!inkling_mlp_forward(context, layer, context->norm, tokens, context->block)) return 0;
    if (!convolve(context, context->block, tokens, hidden, layer->mlp_conv))
        return failure("MLP output convolution failed");
    trace_values(context, "mlp_conv", context->block, count);
    for (size_t i = 0; i < count; i++) output[i] += context->block[i];
    if (!finite_values(output, count)) return failure("MLP residual overflow");
    trace_values(context, "output", output, count);
    return 1;
}

static int forward_rows(InklingContext *context, const uint32_t *ids, size_t tokens,
                        float *logits, size_t logits_count, int last)
{
    if (context == NULL) return failure("missing context");
    inkling_context_reset(context);
    const InklingModel *model = context->model;
    const InklingConfig *config = &model->config;
    size_t count, bytes, id_bytes;
    if (!model->complete || ids == NULL || logits == NULL || tokens == 0 ||
        tokens > context->capacity || tokens > config->model_max_length ||
        !inkling_size_mul(last ? 1 : tokens, config->unpadded_vocab_size, &count) || logits_count < count ||
        !inkling_size_mul(count, sizeof(float), &bytes) || bytes > (size_t)PTRDIFF_MAX ||
        !inkling_size_mul(tokens, sizeof(uint32_t), &id_bytes) ||
        overlap(ids, id_bytes, logits, bytes))
        return failure("invalid prefix, output capacity, overlapping buffers or incomplete model");
    for (size_t t = 0; t < tokens; t++)
        if (ids[t] >= config->vocab_size) return failure("token ID exceeds stored vocabulary");
    if (!scratch_reserve(context, tokens)) return failure("cannot allocate reference scratch");
    size_t hidden = config->hidden_size;
    for (size_t t = 0; t < tokens; t++) {
        size_t row_bytes = hidden * sizeof(uint16_t);
        if (row_bytes > context->weight_bytes ||
            !inkling_context_read(context, model->embed.data, (uint64_t)ids[t] * row_bytes,
                                row_bytes, context->weights)) return failure("embedding read failed");
        float *x = context->hidden[0] + t * hidden;
        for (size_t j = 0; j < hidden; j++) {
            uint16_t value = (uint16_t)((uint16_t)context->weights[2 * j] |
                                      ((uint16_t)context->weights[2 * j + 1] << 8));
            x[j] = inkling_bf16_to_f32(value);
        }
        if (!inkling_rmsnorm(x, hidden, model->embed_norm, hidden,
                             config->rms_norm_epsilon, x, hidden)) return failure("embedding norm failed");
    }
    trace_values(context, "embedding", context->hidden[0], tokens * hidden);
    size_t current = 0;
    for (size_t l = 0; l < model->layer_count; l++) {
        if (!inkling_layer_forward(context, &model->layers[l], context->hidden[current],
                                   tokens, context->hidden[1 - current])) return 0;
        current = 1 - current;
    }
    for (size_t t = last ? tokens - 1 : 0; t < tokens; t++) {
        float *x = context->hidden[current] + t * hidden;
        if (!inkling_rmsnorm(x, hidden, model->final_norm, hidden,
                             config->rms_norm_epsilon, x, hidden)) return failure("final norm failed");
        trace_values(context, "final_norm", x, hidden);
        for (size_t j = 0; j < hidden; j++) x[j] /= config->logits_mup_width_multiplier;
        if (!inkling_project(context, &model->unembed, 0, config->unpadded_vocab_size, x,
                     logits + (last ? 0 : t) * config->unpadded_vocab_size)) return failure("unembedding failed");
    }
    trace_values(context, "logits", logits, count);
    context->stats.tokens_processed = tokens > UINT64_MAX - context->stats.tokens_processed
        ? UINT64_MAX : context->stats.tokens_processed + tokens;
    return 1;
}

int inkling_forward(InklingContext *context, const uint32_t *ids, size_t tokens,
                    float *logits, size_t logits_count)
{
    return forward_rows(context, ids, tokens, logits, logits_count, 0);
}

int inkling_forward_last(InklingContext *context, const uint32_t *ids, size_t tokens,
                         float *logits, size_t logits_count)
{
    return forward_rows(context, ids, tokens, logits, logits_count, 1);
}

static int token_attention(InklingContext *context, const InklingLayer *layer,
                            InklingLayerState *state, size_t position)
{
    const InklingConfig *config = &context->model->config;
    size_t hidden = config->hidden_size, q_width = layer->q.rows;
    size_t kv_width = layer->k.rows, relative_width = layer->r.rows;
    size_t dimension = layer->head_dim, relative = config->relative_dimension;
    if (!inkling_project(context, &layer->q, 0, q_width, context->norm, context->q) ||
        !inkling_project(context, &layer->k, 0, kv_width, context->norm, context->k) ||
        !inkling_project(context, &layer->v, 0, kv_width, context->norm, context->v) ||
        !inkling_project(context, &layer->r, 0, relative_width, context->norm, context->relative))
        return failure("incremental attention projection failed");
    trace_values(context, "q_proj", context->q, q_width);
    trace_values(context, "k_proj", context->k, kv_width);
    trace_values(context, "v_proj", context->v, kv_width);
    trace_values(context, "r_proj", context->relative, relative_width);
    if (!inkling_conv4(context->k, kv_width, 1, kv_width, layer->k_conv, kv_width * 4,
                       state->key_history, kv_width * 3, context->k, kv_width) ||
        !inkling_conv4(context->v, kv_width, 1, kv_width, layer->v_conv, kv_width * 4,
                       state->value_history, kv_width * 3, context->v, kv_width))
        return failure("incremental key/value convolution failed");
    trace_values(context, "k_conv", context->k, kv_width);
    trace_values(context, "v_conv", context->v, kv_width);
    for (size_t h = 0; h < layer->heads; h++) {
        float *q = context->q + h * dimension;
        if (!inkling_rmsnorm(q, dimension, layer->q_norm, dimension,
                             config->rms_norm_epsilon, q, dimension)) return 0;
    }
    for (size_t h = 0; h < layer->kv_heads; h++) {
        float *k = context->k + h * dimension;
        if (!inkling_rmsnorm(k, dimension, layer->k_norm, dimension,
                             config->rms_norm_epsilon, k, dimension)) return 0;
    }
    trace_values(context, "q_norm", context->q, q_width);
    trace_values(context, "k_norm", context->k, kv_width);
    size_t slot = position % state->capacity;
    memcpy(state->keys + slot * kv_width, context->k, kv_width * sizeof(float));
    memcpy(state->values + slot * kv_width, context->v, kv_width * sizeof(float));
    float tau = 1.0f;
    if (layer->window == 0) {
        float ratio = (float)(position + 1) / (float)config->log_scaling_n_floor;
        tau += config->log_scaling_alpha * logf(fmaxf(1.0f, ratio));
    }
    if (!isfinite(tau)) return failure("incremental attention scaling overflow");
    for (size_t j = 0; j < q_width; j++) context->q[j] *= tau;
    if (!finite_values(context->q, q_width)) return failure("incremental query scaling overflow");
    trace_values(context, "query_scaled", context->q, q_width);
    size_t start = position + 1 > state->capacity ? position + 1 - state->capacity : 0;
    for (size_t h = 0; h < layer->heads; h++) {
        const float *q = context->q + h * dimension;
        const float *r = context->relative + h * relative;
        size_t kv = h / (layer->heads / layer->kv_heads);
        for (size_t s = 0; s <= position; s++) {
            float bias = 0;
            if (position - s < layer->extent)
                for (size_t j = 0; j < relative; j++)
                    bias += r[j] * layer->relative[j * layer->extent + position - s];
            context->bias[s] = bias;
            if (s < start) {
                context->scores[s] = -INFINITY;
                continue;
            }
            const float *k = state->keys + (s % state->capacity) * kv_width + kv * dimension;
            float score = 0;
            for (size_t j = 0; j < dimension; j++) score += q[j] * k[j];
            context->scores[s] = score / (float)dimension + bias * tau;
        }
        if (!finite_values(context->bias, position + 1)) return failure("incremental relative bias overflow");
        trace_values(context, "relative_bias", context->bias, position + 1);
        for (size_t s = 0; s <= position; s++) context->bias[s] *= tau;
        if (!finite_values(context->bias, position + 1)) return failure("incremental relative scaling overflow");
        trace_values(context, "relative_scaled", context->bias, position + 1);
        trace_values(context, "attention_scores", context->scores, position + 1);
        if (!inkling_softmax(context->scores, position + 1, context->scores, position + 1))
            return failure("incremental attention softmax failed");
        trace_values(context, "attention_probs", context->scores, position + 1);
        float *head = context->heads + h * dimension;
        for (size_t j = 0; j < dimension; j++) {
            float value = 0;
            for (size_t s = start; s <= position; s++)
                value += context->scores[s] * state->values[(s % state->capacity) * kv_width + kv * dimension + j];
            head[j] = value;
        }
    }
    trace_values(context, "attention_heads", context->heads, q_width);
    if (!inkling_project(context, &layer->o, 0, hidden, context->heads, context->block))
        return failure("incremental attention output projection failed");
    trace_values(context, "attention_output", context->block, hidden);
    return 1;
}

static int token_layer(InklingContext *context, const InklingLayer *layer,
                        InklingLayerState *state, size_t position,
                        const float *input, float *output)
{
    const InklingConfig *config = &context->model->config;
    size_t hidden = config->hidden_size;
    if (!inkling_rmsnorm(input, hidden, layer->attn_norm, hidden,
                         config->rms_norm_epsilon, context->norm, hidden)) return 0;
    trace_values(context, "attn_norm", context->norm, hidden);
    if (!token_attention(context, layer, state, position) ||
        !inkling_conv4(context->block, hidden, 1, hidden, layer->attn_conv, hidden * 4,
                       state->attn_history, hidden * 3, context->block, hidden)) return 0;
    trace_values(context, "attn_conv", context->block, hidden);
    for (size_t i = 0; i < hidden; i++) output[i] = input[i] + context->block[i];
    if (!finite_values(output, hidden)) return failure("incremental attention residual overflow");
    trace_values(context, "post_attention", output, hidden);
    if (!inkling_rmsnorm(output, hidden, layer->mlp_norm, hidden,
                         config->rms_norm_epsilon, context->norm, hidden)) return 0;
    trace_values(context, "mlp_norm", context->norm, hidden);
    if (!inkling_mlp_forward(context, layer, context->norm, 1, context->block) ||
        !inkling_conv4(context->block, hidden, 1, hidden, layer->mlp_conv, hidden * 4,
                       state->mlp_history, hidden * 3, context->block, hidden)) return 0;
    trace_values(context, "mlp_conv", context->block, hidden);
    for (size_t i = 0; i < hidden; i++) output[i] += context->block[i];
    if (!finite_values(output, hidden)) return failure("incremental MLP residual overflow");
    trace_values(context, "output", output, hidden);
    return 1;
}

static int token_forward(InklingContext *context, uint32_t id, float *logits)
{
    const InklingModel *model = context->model;
    const InklingConfig *config = &model->config;
    size_t hidden = config->hidden_size, row_bytes = hidden * sizeof(uint16_t);
    if (!inkling_context_read(context, model->embed.data, (uint64_t)id * row_bytes,
                              row_bytes, context->weights)) return failure("incremental embedding read failed");
    float *x = context->hidden[0];
    for (size_t j = 0; j < hidden; j++) {
        uint16_t value = (uint16_t)((uint16_t)context->weights[2 * j] |
                                  ((uint16_t)context->weights[2 * j + 1] << 8));
        x[j] = inkling_bf16_to_f32(value);
    }
    if (!inkling_rmsnorm(x, hidden, model->embed_norm, hidden,
                         config->rms_norm_epsilon, x, hidden)) return 0;
    trace_values(context, "embedding", x, hidden);
    size_t current = 0;
    for (size_t i = 0; i < model->layer_count; i++) {
        if (!token_layer(context, &model->layers[i], &context->sequence->layers[i],
                         context->sequence->position, context->hidden[current],
                         context->hidden[1 - current])) return 0;
        current = 1 - current;
    }
    if (logits != NULL) {
        x = context->hidden[current];
        if (!inkling_rmsnorm(x, hidden, model->final_norm, hidden,
                             config->rms_norm_epsilon, x, hidden)) return 0;
        trace_values(context, "final_norm", x, hidden);
        for (size_t j = 0; j < hidden; j++) x[j] /= config->logits_mup_width_multiplier;
        if (!inkling_project(context, &model->unembed, 0, config->unpadded_vocab_size, x, logits))
            return failure("incremental unembedding failed");
        trace_values(context, "logits", logits, config->unpadded_vocab_size);
    }
    context->sequence->position++;
    if (context->stats.tokens_processed != UINT64_MAX) context->stats.tokens_processed++;
    return 1;
}

static int sequence_failure(InklingContext *context, const char *message)
{
    if (context != NULL && context->sequence != NULL) {
        context->sequence->valid = 0;
        context->sequence->position = 0;
    }
    return failure(message);
}

size_t inkling_context_position(const InklingContext *context)
{
    return context != NULL && context->sequence != NULL && context->sequence->valid
        ? context->sequence->position : 0;
}

int inkling_prefill(InklingContext *context, const uint32_t *ids, size_t tokens,
                     float *logits, size_t logits_count)
{
    if (context == NULL) return failure("missing incremental context");
    inkling_context_reset(context);
    size_t bytes, id_bytes;
    if (!context->incremental || context->sequence == NULL || !context->model->complete ||
        ids == NULL || logits == NULL || tokens == 0 || tokens > context->capacity ||
        tokens > context->model->config.model_max_length ||
        logits_count < context->model->config.unpadded_vocab_size ||
        !inkling_size_mul(context->model->config.unpadded_vocab_size, sizeof(float), &bytes) ||
        !inkling_size_mul(tokens, sizeof(*ids), &id_bytes) || overlap(ids, id_bytes, logits, bytes))
        return sequence_failure(context, "invalid prefill buffers, tokens or context");
    for (size_t i = 0; i < tokens; i++)
        if (ids[i] >= context->model->config.vocab_size)
            return sequence_failure(context, "prefill token ID exceeds stored vocabulary");
    for (size_t i = 0; i < tokens; i++)
        if (!token_forward(context, ids[i], i + 1 == tokens ? logits : NULL))
            return sequence_failure(context, "prefill failed");
    return 1;
}

int inkling_decode(InklingContext *context, uint32_t id, float *logits, size_t logits_count)
{
    if (context == NULL || !context->incremental || context->sequence == NULL ||
        !context->sequence->valid || !context->model->complete || logits == NULL ||
        logits_count < context->model->config.unpadded_vocab_size ||
        id >= context->model->config.vocab_size ||
        context->sequence->position >= context->capacity ||
        context->sequence->position >= context->model->config.model_max_length)
        return sequence_failure(context, "invalid decode token, output, sequence or capacity");
    if (!token_forward(context, id, logits)) return sequence_failure(context, "decode failed");
    return 1;
}
