#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "inkling/inkling.h"
#include "inkling_json.h"

#define MAX_CONFIG_BYTES (1024L * 1024L)

static char *read_text_file(const char *path, size_t *output_length)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(stderr, "cannot open config: %s\n", path);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }

    long length = ftell(file);

    if (length < 0 || length > MAX_CONFIG_BYTES) {
        fprintf(stderr, "invalid config file size\n");
        fclose(file);
        return NULL;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    char *text = malloc((size_t)length + 1);

    if (text == NULL) {
        fprintf(stderr, "cannot allocate config buffer\n");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(text, 1, (size_t)length, file);
    fclose(file);

    if (bytes_read != (size_t)length) {
        fprintf(stderr, "could not read complete config\n");
        free(text);
        return NULL;
    }

    text[(size_t)length] = '\0';
    *output_length = (size_t)length;
    return text;
}

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

    float converted = (float)number;

    if (!isfinite(converted)) {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    *output = converted;
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

int inkling_config_load(
    const char *path,
    InklingConfig *config
)
{
    if (path == NULL || config == NULL) {
        return 0;
    }

    size_t json_length = 0;
    char *json = read_text_file(path, &json_length);

    if (json == NULL) {
        return 0;
    }

    InklingJsonDocument document = {0};
    InklingJsonError error;

    if (!inkling_json_parse(
            json,
            json_length,
            &document,
            &error)) {
        fprintf(
            stderr,
            "invalid config JSON at %zu:%zu: %s\n",
            error.line,
            error.column,
            error.message
        );
        free(json);
        return 0;
    }

    free(json);

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

    if (config->sliding_window_size >
        config->model_max_length) {
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
