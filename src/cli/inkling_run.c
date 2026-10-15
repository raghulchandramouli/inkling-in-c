#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inkling/inkling.h"
#include "inkling/inkling_model.h"
#include "../core/inkling_alloc.h"

static int parse_ids(const char *text, uint32_t **ids, size_t *count)
{
    size_t length = 1;
    for (const char *p = text; *p; p++) {
        if (*p == ',' && !inkling_size_add(length, 1, &length)) return 0;
    }
    if (!inkling_alloc_array(length, sizeof(**ids), (void **)ids)) return 0;
    const char *p = text;
    for (size_t i = 0; i < length; i++) {
        if (*p < '0' || *p > '9') goto invalid;
        char *end;
        errno = 0;
        unsigned long long value = strtoull(p, &end, 10);
        if (errno || value > UINT32_MAX || (i + 1 == length ? *end != '\0' : *end != ','))
            goto invalid;
        (*ids)[i] = (uint32_t)value;
        p = end + (i + 1 < length ? 1 : 0);
    }
    *count = length;
    return 1;
invalid:
    free(*ids);
    *ids = NULL;
    return 0;
}

static int write_logits(const char *path, const float *logits, size_t count,
                        size_t tokens, size_t vocab)
{
    size_t bytes;
    char *temporary = NULL;
    if (!inkling_size_add(strlen(path), sizeof(".tmp.XXXXXX"), &bytes) ||
        !inkling_alloc_array(bytes, 1, (void **)&temporary)) return 0;
    (void)snprintf(temporary, bytes, "%s.tmp.XXXXXX", path);
    int fd = mkstemp(temporary);
    if (fd < 0) { free(temporary); return 0; }
    FILE *file = fdopen(fd, "w");
    int success = file != NULL;
    if (file == NULL) (void)close(fd);
    if (success && fprintf(file, "{\"tokens\":%zu,\"vocab\":%zu,\"logits\":[", tokens, vocab) < 0)
        success = 0;
    for (size_t i = 0; success && i < count; i++)
        if (!isfinite(logits[i]) || fprintf(file, "%s%.9g", i ? "," : "", (double)logits[i]) < 0)
            success = 0;
    if (success && fputs("]}\n", file) == EOF) success = 0;
    if (file != NULL && fclose(file) != 0) success = 0;
    if (success && rename(temporary, path) != 0) success = 0;
    if (!success) (void)unlink(temporary);
    free(temporary);
    return success;
}

static int run_ids(int argc, char **argv)
{
    uint32_t *ids = NULL;
    size_t tokens = 0, count;
    if (argc != 6 || strcmp(argv[4], "--dump-logits") != 0 ||
        !parse_ids(argv[3], &ids, &tokens)) {
        fputs("expected MODEL_DIR --ids ID,ID,... --dump-logits PATH\n", stderr);
        return 2;
    }
    InklingModel *model = NULL;
    InklingContext *context = NULL;
    float *logits = NULL;
    int success = inkling_model_open(argv[1], &model) &&
        inkling_size_mul(tokens, inkling_model_logits_width(model), &count) &&
        inkling_alloc_array(count, sizeof(*logits), (void **)&logits) &&
        inkling_context_create(model, tokens, &context) &&
        inkling_forward(context, ids, tokens, logits, count) &&
        write_logits(argv[5], logits, count, tokens, inkling_model_logits_width(model));
    if (!success) fputs("raw token forward failed\n", stderr);
    free(ids);
    free(logits);
    inkling_context_free(context);
    inkling_model_close(model);
    return success ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && strcmp(argv[2], "--ids") == 0) return run_ids(argc, argv);
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
                "       %s MODEL_DIR --verify-model [--with-mtp] [--metadata-only]\n"
                "       %s MODEL_DIR --ids ID,ID,... --dump-logits PATH\n",
                argv[0], argv[0], argv[0]);
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
