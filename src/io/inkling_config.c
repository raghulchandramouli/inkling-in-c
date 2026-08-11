#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

#define MAX_CONFIG_BYTES (1024L * 1024L)

static char *read_text_file(const char *path)
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

    text[length] = '\0';
    return text;
}

static const char *find_value(const char *json, const char *key)
{
    char pattern[96];

    int length = snprintf(
        pattern,
        sizeof(pattern),
        "\"%s\"",
        key
    );

    if (length < 0 || (size_t)length >= sizeof(pattern)) {
        return NULL;
    }

    const char *position = strstr(json, pattern);

    if (position == NULL) {
        return NULL;
    }

    position += length;

    while (isspace((unsigned char)*position)) {
        position++;
    }

    if (*position != ':') {
        return NULL;
    }

    position++;

    while (isspace((unsigned char)*position)) {
        position++;
    }

    return position;
}

static int number_ends_here(const char *position)
{
    while (isspace((unsigned char)*position)) {
        position++;
    }

    return *position == ',' ||
           *position == '}' ||
           *position == ']';
}

static int read_u32(
    const char *json,
    const char *key,
    uint32_t *output
)
{
    const char *start = find_value(json, key);

    if (start == NULL || *start == '-') {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    errno = 0;

    char *end = NULL;
    unsigned long value = strtoul(start, &end, 10);

    if (errno != 0 ||
        end == start ||
        value > UINT32_MAX ||
        !number_ends_here(end)) {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    *output = (uint32_t)value;
    return 1;
}

static int read_float(
    const char *json,
    const char *key,
    float *output
)
{
    const char *start = find_value(json, key);

    if (start == NULL) {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    errno = 0;

    char *end = NULL;
    float value = strtof(start, &end);

    if (errno != 0 ||
        end == start ||
        !isfinite(value) ||
        !number_ends_here(end)) {
        fprintf(stderr, "missing or invalid field: %s\n", key);
        return 0;
    }

    *output = value;
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

    char *json = read_text_file(path);

    if (json == NULL) {
        return 0;
    }

    InklingConfig parsed = {0};

    /*
     * local_layer_ids contains five local layers followed by
     * one global layer, producing a six-layer repeating pattern.
     */
    parsed.global_attention_stride = 6;

    int success =
        read_u32(json, "model_max_length",
                 &parsed.model_max_length) &&
        read_u32(json, "vocab_size",
                 &parsed.vocab_size) &&
        read_u32(json, "eos_token_id",
                 &parsed.eos_token_id) &&
        read_u32(json, "hidden_size",
                 &parsed.hidden_size) &&
        read_u32(json, "num_hidden_layers",
                 &parsed.num_hidden_layers) &&
        read_u32(json, "num_attention_heads",
                 &parsed.num_attention_heads) &&
        read_u32(json, "num_key_value_heads",
                 &parsed.num_key_value_heads) &&
        read_u32(json, "head_dim",
                 &parsed.head_dim) &&
        read_u32(json, "sliding_window_size",
                 &parsed.sliding_window_size) &&
        read_u32(json, "d_rel",
                 &parsed.relative_dimension) &&
        read_u32(json, "rel_extent",
                 &parsed.relative_extent) &&
        read_u32(json, "sconv_kernel_size",
                 &parsed.sconv_kernel_size) &&
        read_u32(json, "n_routed_experts",
                 &parsed.num_routed_experts) &&
        read_u32(json, "num_experts_per_tok",
                 &parsed.num_experts_per_token) &&
        read_u32(json, "n_shared_experts",
                 &parsed.num_shared_experts) &&
        read_u32(json, "dense_intermediate_size",
                 &parsed.dense_intermediate_size) &&
        read_u32(json, "intermediate_size",
                 &parsed.expert_intermediate_size) &&
        read_float(json, "rms_norm_eps",
                   &parsed.rms_norm_epsilon) &&
        read_float(json, "route_scale",
                   &parsed.route_scale);

    free(json);

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
