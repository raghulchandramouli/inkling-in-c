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

static int read_global_attention_stride(
    const InklingJsonValue *text_config,
    uint32_t num_hidden_layers,
    uint32_t *output
)
{
    const InklingJsonValue *value =
        inkling_json_object_get(text_config, "local_layer_ids");
    size_t count = inkling_json_array_size(value);

    if (inkling_json_type(value) != INKLING_JSON_ARRAY ||
        count >= num_hidden_layers) {
        fputs("missing or invalid field: local_layer_ids\n", stderr);
        return 0;
    }

    unsigned char *local = calloc(num_hidden_layers, sizeof(*local));

    if (local == NULL) {
        fputs("cannot allocate local-layer map\n", stderr);
        return 0;
    }

    int success = 1;

    for (size_t index = 0; index < count; index++) {
        uint64_t layer = 0;

        if (!inkling_json_number_u64(
                inkling_json_array_at(value, index),
                &layer) ||
            layer >= num_hidden_layers ||
            local[(size_t)layer]) {
            success = 0;
            break;
        }

        local[(size_t)layer] = 1;
    }

    uint32_t stride = 0;

    if (success) {
        for (uint32_t layer = 0; layer < num_hidden_layers; layer++) {
            if (!local[layer]) {
                stride = layer + 1;
                break;
            }
        }
    }

    if (stride == 0) {
        success = 0;
    }

    if (success) {
        for (uint32_t layer = 0; layer < num_hidden_layers; layer++) {
            int expected_local = (layer + 1) % stride != 0;

            if ((local[layer] != 0) != expected_local) {
                success = 0;
                break;
            }
        }
    }

    free(local);

    if (!success) {
        fputs("local_layer_ids is not a regular local/global pattern\n", stderr);
        return 0;
    }

    *output = stride;
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
        read_global_attention_stride(
            text_config,
            parsed.num_hidden_layers,
            &parsed.global_attention_stride
        );

    inkling_json_document_free(&document);

    if (!success || !inkling_config_is_valid(&parsed)) {
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

    if (config->rms_norm_epsilon <= 0.0f ||
        config->route_scale <= 0.0f) {
        return 0;
    }

    return 1;
}
