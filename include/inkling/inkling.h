#ifndef INKLING_INKLING_H
#define INKLING_INKLING_H

#include <stdint.h>

typedef struct {
    uint32_t model_max_length;
    uint32_t vocab_size;
    uint32_t eos_token_id;

    uint32_t hidden_size;
    uint32_t num_hidden_layers;

    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;

    uint32_t sliding_window_size;
    uint32_t global_attention_stride;
    uint32_t relative_dimension;
    uint32_t relative_extent;
    uint32_t sconv_kernel_size;

    uint32_t num_routed_experts;
    uint32_t num_experts_per_token;
    uint32_t num_shared_experts;

    uint32_t dense_intermediate_size;
    uint32_t expert_intermediate_size;

    float rms_norm_epsilon;
    float route_scale;
} InklingConfig;

int inkling_config_load(const char *path, InklingConfig *config);
int inkling_config_is_valid(const InklingConfig *config);

int inkling_safetensors_header_size(
    const char *path,
    uint64_t *header_size
);

/*
* On success, the caller owns *header_json and must free it.
*/

int inkling_safetensors_read_header(
    const char *path,
    char **header_json,
    uint64_t *header_size
);

#endif 