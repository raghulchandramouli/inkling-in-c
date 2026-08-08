#include "inkling/inkling.h"

InklingConfig inkling_small_config(void)
{
    return (InklingConfig) {
        .model_max_length = 1048576,
        .vocab_size = 201024,
        .eos_token_id = 200006,

        .hidden_size = 4096,
        .num_hidden_layers = 42,

        .num_attention_heads = 32,
        .num_key_value_heads = 8,
        .head_dim = 128,

        .sliding_window_size = 512,
        .global_attention_stride = 6,
        .relative_dimension = 16,
        .relative_extent = 1024,
        .sconv_kernel_size = 4,

        .num_routed_experts = 256,
        .num_experts_per_token = 6,
        .num_shared_experts = 2,

        .dense_intermediate_size = 16384,
        .expert_intermediate_size = 2048,

        .rms_norm_epsilon = 1.0e-6f,
        .route_scale = 8.0f,
    };
}

int inkling_config_is_valid(const InklingConfig *config)
{
    if (config == 0) {
        return 0;
    }

    if (config->hidden_size == 0 ||
        config->num_hidden_layers == 0 ||
        config->num_attention_heads == 0 ||
        config->num_key_value_heads == 0 ||
        config->head_dim == 0) {
        return 0;
    }

    if (config->hidden_size % config->num_attention_heads != 0 ||
        config->hidden_size / config->num_attention_heads != config->head_dim) {
        return 0;
    }

    if (config->num_attention_heads % config->num_key_value_heads != 0) {
        return 0;
    }

    if (config->num_experts_per_token > config->num_routed_experts) {
        return 0;
    }

    if (config->eos_token_id >= config->vocab_size) {
        return 0;
    }

    if (config->sliding_window_size > config->model_max_length) {
        return 0;
    }

    if (config->rms_norm_epsilon <= 0.0f ||
        config->route_scale <= 0.0f) {
        return 0;
    }

    return 1;
}