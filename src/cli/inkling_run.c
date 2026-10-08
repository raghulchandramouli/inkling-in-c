#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[2], "--verify-model") == 0) {
        int with_mtp = 0, metadata_only = 0;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--with-mtp") == 0 && !with_mtp) {
                with_mtp = 1;
            } else if (strcmp(argv[i], "--metadata-only") == 0 && !metadata_only) {
                metadata_only = 1;
            } else {
                fprintf(stderr, "unknown or duplicate verification option: %s\n", argv[i]);
                return 2;
            }
        }
        return inkling_verify_model(argv[1], with_mtp, metadata_only) ? 0 : 3;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s <config.json>\n"
                "       %s MODEL_DIR --verify-model [--with-mtp] [--metadata-only]\n",
                argv[0], argv[0]);
        return 2;
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

    printf("local layers:        %zu [", config.num_local_layers);
    for (size_t index = 0; index < config.num_local_layers; index++) {
        printf("%s%" PRIu32, index == 0 ? "" : ",", config.local_layer_ids[index]);
    }
    puts("]");
    printf("global layers:       %zu\n",
           (size_t)config.num_hidden_layers - config.num_local_layers);

    printf("routed experts:      %" PRIu32 " of %" PRIu32 "\n",
           config.num_experts_per_token,
           config.num_routed_experts);

    printf("shared experts:      %" PRIu32 "\n",
           config.num_shared_experts);

    printf("vocabulary:          %" PRIu32 "\n",
           config.vocab_size);

    printf("maximum context:     %" PRIu32 " tokens\n",
           config.model_max_length);

    inkling_config_free(&config);
    return EXIT_SUCCESS;
}
