#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "inkling/inkling.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <config.json>\n", argv[0]);
        return EXIT_FAILURE;
    }

    InklingConfig config;

    if (!inkling_config_load(argv[1], &config)) {
        fputs("failed to load Inkling configuration\n", stderr);
        return EXIT_FAILURE;
    }

    puts("Inkling-Small configuration OK");

    printf("layers:              %" PRIu32 "\n",
           config.num_hidden_layers);

    printf("hidden size:         %" PRIu32 "\n",
           config.hidden_size);

    printf("attention:           %" PRIu32 " heads x %" PRIu32 "\n",
           config.num_attention_heads,
           config.head_dim);

    printf("key/value heads:     %" PRIu32 "\n",
           config.num_key_value_heads);

    printf("local window:        %" PRIu32 " tokens\n",
           config.sliding_window_size);

    printf("global attention:    every %" PRIu32 "th layer\n",
           config.global_attention_stride);

    printf("routed experts:      %" PRIu32 " of %" PRIu32 "\n",
           config.num_experts_per_token,
           config.num_routed_experts);

    printf("shared experts:      %" PRIu32 "\n",
           config.num_shared_experts);

    printf("vocabulary:          %" PRIu32 "\n",
           config.vocab_size);

    printf("maximum context:     %" PRIu32 " tokens\n",
           config.model_max_length);

    return EXIT_SUCCESS;
}