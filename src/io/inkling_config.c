#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"
#include "inkling_json.h"

#define MAX_CONFIG_BYTES (1024L * 1024L)

static int read_u32(
    const InklingJsonValue *object,
    const char *key,
    uint32_t *output
)
{
    const InklingJsonValue *value =
        inkling_json_object_get(object, key);
    uint64_t number = 0;

    if (!inkling_json_number_u64(value, &number) ||
        number > UINT32_MAX) {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    *output = (uint32_t)number;
    return 1;
}

static int read_float(
    const InklingJsonValue *object,
    const char *key,
    float *output
)
{
    const InklingJsonValue *value =
        inkling_json_object_get(object, key);
    double number = 0.0;

    if (!inkling_json_number_double(value, &number) ||
        number < -(double)FLT_MAX ||
        number > (double)FLT_MAX) {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    *output = (float)number;
    return 1;
}

static int read_local_layer_ids(
    const InklingJsonValue *text_config,
    InklingConfig *config
)
{
    const InklingJsonValue *value =
        inkling_json_object_get(text_config, "local_layer_ids");
    size_t count = inkling_json_array_size(value);

    if (inkling_json_type(value) != INKLING_JSON_ARRAY ||
        count > config->num_hidden_layers ||
        count > SIZE_MAX / sizeof(*config->local_layer_ids)) {
        fputs("missing or invalid field: local_layer_ids\n", stderr);
        return 0;
    }

    if (count == 0) {
        return 1;
    }

    config->local_layer_ids = malloc(count * sizeof(*config->local_layer_ids));

    if (config->local_layer_ids == NULL) {
        fputs("cannot allocate local-layer list\n", stderr);
        return 0;
    }

    config->num_local_layers = count;

    for (size_t index = 0; index < count; index++) {
        uint64_t layer = 0;

        if (!inkling_json_number_u64(
                inkling_json_array_at(value, index),
                &layer) ||
            layer >= config->num_hidden_layers) {
            fputs("invalid local layer ID\n", stderr);
            return 0;
        }

        config->local_layer_ids[index] = (uint32_t)layer;
    }

    return 1;
}

static int execution_config(const InklingJsonValue *text, InklingConfig *config)
{
    struct { const char *key; uint32_t *out; } integers[] = {
        {"swa_head_dim", &config->swa_head_dim},
        {"swa_num_attention_heads", &config->swa_num_attention_heads},
        {"swa_num_key_value_heads", &config->swa_num_key_value_heads},
        {"dense_mlp_idx", &config->dense_mlp_idx},
        {"unpadded_vocab_size", &config->unpadded_vocab_size},
        {"log_scaling_n_floor", &config->log_scaling_n_floor}
    };
    for (size_t i = 0; i < sizeof(integers) / sizeof(integers[0]); i++) {
        uint64_t number;
        if (!inkling_json_number_u64(inkling_json_object_get(text, integers[i].key), &number) ||
            number > UINT32_MAX) return 0;
        *integers[i].out = (uint32_t)number;
    }
    struct { const char *key; float *out; } numbers[] = {
        {"log_scaling_alpha", &config->log_scaling_alpha},
        {"logits_mup_width_multiplier", &config->logits_mup_width_multiplier}
    };
    for (size_t i = 0; i < sizeof(numbers) / sizeof(numbers[0]); i++) {
        double number;
        if (!inkling_json_number_double(inkling_json_object_get(text, numbers[i].key), &number) ||
            number < 0 || number > FLT_MAX) return 0;
        *numbers[i].out = (float)number;
    }
    static const struct { const char *key; int expected; } flags[] = {
        {"q_bias", 0}, {"o_bias", 0}, {"use_embed_norm", 1}, {"use_sconv", 1},
        {"shared_expert_sink", 1}, {"use_gate_bias", 1}, {"norm_after_topk", 1},
        {"use_global_scale", 1}
    };
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++) {
        int flag;
        if (!inkling_json_boolean(inkling_json_object_get(text, flags[i].key), &flag) ||
            flag != flags[i].expected) {
            fprintf(stderr, "unsupported text execution field: %s (expected %s)\n",
                    flags[i].key, flags[i].expected ? "true" : "false");
            return 0;
        }
    }
    size_t length;
    const char *activation = inkling_json_string(
        inkling_json_object_get(text, "gate_activation"), &length);
    const InklingJsonValue *softcap = inkling_json_object_get(text, "final_logit_softcapping");
    return activation && length == 7 && memcmp(activation, "sigmoid", 7) == 0 &&
        softcap && inkling_json_type(softcap) == INKLING_JSON_NULL &&
        config->swa_head_dim && config->swa_num_attention_heads &&
        config->swa_num_key_value_heads &&
        config->swa_num_attention_heads % config->swa_num_key_value_heads == 0 &&
        (uint64_t)config->swa_num_attention_heads * config->swa_head_dim <= SIZE_MAX &&
        config->dense_mlp_idx <= config->num_hidden_layers &&
        config->unpadded_vocab_size && config->unpadded_vocab_size <= config->vocab_size &&
        config->log_scaling_n_floor && config->logits_mup_width_multiplier > 0 &&
        config->sconv_kernel_size == 4 && config->num_shared_experts > 0;
}

int inkling_config_load(
    const char *path,
    InklingConfig *config
)
{
    if (path == NULL || config == NULL) {
        return 0;
    }

    InklingJsonDocument document = {0};
    if (!inkling_json_load_file(path, MAX_CONFIG_BYTES, &document)) {
        return 0;
    }

    const InklingJsonValue *root = document.root;
    const InklingJsonValue *text_config =
        inkling_json_object_get(root, "text_config");

    if (inkling_json_type(root) != INKLING_JSON_OBJECT ||
        inkling_json_type(text_config) != INKLING_JSON_OBJECT) {
        fputs("config must contain a text_config object\n", stderr);
        inkling_json_document_free(&document);
        return 0;
    }

    InklingConfig parsed = {0};

    int success =
        read_u32(text_config, "model_max_length",
                 &parsed.model_max_length) &&
        read_u32(text_config, "vocab_size",
                 &parsed.vocab_size) &&
        read_u32(root, "eos_token_id",
                 &parsed.eos_token_id) &&
        read_u32(text_config, "hidden_size",
                 &parsed.hidden_size) &&
        read_u32(text_config, "num_hidden_layers",
                 &parsed.num_hidden_layers) &&
        read_u32(text_config, "num_attention_heads",
                 &parsed.num_attention_heads) &&
        read_u32(text_config, "num_key_value_heads",
                 &parsed.num_key_value_heads) &&
        read_u32(text_config, "head_dim",
                 &parsed.head_dim) &&
        read_u32(text_config, "sliding_window_size",
                 &parsed.sliding_window_size) &&
        read_u32(text_config, "d_rel",
                 &parsed.relative_dimension) &&
        read_u32(text_config, "rel_extent",
                 &parsed.relative_extent) &&
        read_u32(text_config, "sconv_kernel_size",
                 &parsed.sconv_kernel_size) &&
        read_u32(text_config, "n_routed_experts",
                 &parsed.num_routed_experts) &&
        read_u32(text_config, "num_experts_per_tok",
                 &parsed.num_experts_per_token) &&
        read_u32(text_config, "n_shared_experts",
                 &parsed.num_shared_experts) &&
        read_u32(text_config, "dense_intermediate_size",
                 &parsed.dense_intermediate_size) &&
        read_u32(text_config, "intermediate_size",
                 &parsed.expert_intermediate_size) &&
        read_float(text_config, "rms_norm_eps",
                   &parsed.rms_norm_epsilon) &&
        read_float(text_config, "route_scale",
                   &parsed.route_scale) &&
        read_local_layer_ids(text_config, &parsed);

    if (success) parsed.execution_fields_valid = execution_config(text_config, &parsed);
    inkling_json_document_free(&document);

    if (!success || !inkling_config_is_valid(&parsed)) {
        inkling_config_free(&parsed);
        return 0;
    }

    *config = parsed;
    return 1;
}

int inkling_config_is_valid(const InklingConfig *config)
{
    if (config == NULL) {
        return 0;
    }

    if (config->model_max_length == 0 ||
        config->vocab_size == 0 ||
        config->hidden_size == 0 ||
        config->num_hidden_layers == 0 ||
        config->num_attention_heads == 0 ||
        config->num_key_value_heads == 0 ||
        config->head_dim == 0 ||
        config->relative_dimension == 0 ||
        config->relative_extent == 0 ||
        config->sconv_kernel_size == 0 ||
        config->dense_intermediate_size == 0 ||
        config->expert_intermediate_size == 0 ||
        config->num_routed_experts == 0 ||
        config->num_experts_per_token == 0) {
        return 0;
    }

    if (config->hidden_size % config->num_attention_heads != 0 ||
        config->hidden_size / config->num_attention_heads !=
            config->head_dim) {
        return 0;
    }

    if (config->num_attention_heads %
        config->num_key_value_heads != 0) {
        return 0;
    }

    if (config->num_experts_per_token >
        config->num_routed_experts) {
        return 0;
    }

    if (config->eos_token_id >= config->vocab_size) {
        return 0;
    }

    if (config->sliding_window_size > config->model_max_length ||
        (config->num_local_layers != 0 && config->sliding_window_size == 0)) {
        return 0;
    }

    if (config->num_local_layers > config->num_hidden_layers ||
        (config->num_local_layers != 0 && config->local_layer_ids == NULL)) {
        return 0;
    }

    for (size_t index = 0; index < config->num_local_layers; index++) {
        if (config->local_layer_ids[index] >= config->num_hidden_layers) {
            return 0;
        }
        /* ponytail: quadratic duplicate check; use a set if layer counts grow. */
        for (size_t previous = 0; previous < index; previous++) {
            if (config->local_layer_ids[previous] == config->local_layer_ids[index]) {
                return 0;
            }
        }
    }

    if (!isfinite(config->rms_norm_epsilon) ||
        !isfinite(config->route_scale) ||
        config->rms_norm_epsilon <= 0.0f ||
        config->route_scale <= 0.0f) {
        return 0;
    }

    return 1;
}

void inkling_config_free(InklingConfig *config)
{
    if (config != NULL) {
        free(config->local_layer_ids);
        config->local_layer_ids = NULL;
        config->num_local_layers = 0;
    }
}
