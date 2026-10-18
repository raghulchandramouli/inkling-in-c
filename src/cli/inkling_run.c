#define _DARWIN_C_SOURCE
#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

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

typedef struct {
    const char *path;
    char *temporary;
    FILE *file;
    int published;
} OutputFile;

static void discard_file(OutputFile *output)
{
    if (output->file != NULL) (void)fclose(output->file);
    if (output->temporary != NULL) {
        (void)unlink(output->temporary);
        free(output->temporary);
    }
}

static int stage_file(OutputFile *output)
{
    size_t bytes;
    void *allocation = NULL;
    if (!inkling_size_add(strlen(output->path), sizeof(".tmp.XXXXXX"), &bytes) ||
        !inkling_alloc_array(bytes, 1, &allocation)) return 0;
    output->temporary = allocation;
    (void)snprintf(output->temporary, bytes, "%s.tmp.XXXXXX", output->path);
    int fd = mkstemp(output->temporary);
    if (fd < 0) return 0;
    output->file = fdopen(fd, "w");
    if (output->file == NULL) { (void)close(fd); return 0; }
    return 1;
}

static int close_stage(OutputFile *output)
{
    int success = fclose(output->file) == 0;
    output->file = NULL;
    return success;
}

static int publish_file(OutputFile *output)
{
    if (rename(output->temporary, output->path) != 0) return 0;
    output->published = 1;
    return 1;
}

static int parse_unsigned(const char *text, uint64_t maximum, uint64_t *out)
{
    if (*text == '\0') return 0;
    uint64_t value = 0;
    for (const char *p = text; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (maximum - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    *out = value;
    return 1;
}

static int parse_decimal(const char *text, double *out)
{
    const char *p = text;
    if (*p == '+' || *p == '-') p++;
    int digits = 0, nonzero = 0;
    while (*p >= '0' && *p <= '9') { digits = 1; nonzero |= *p++ != '0'; }
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') { digits = 1; nonzero |= *p++ != '0'; }
    }
    if (!digits) return 0;
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        const char *start = p;
        while (*p >= '0' && *p <= '9') p++;
        if (p == start) return 0;
    }
    if (*p) return 0;
    errno = 0;
    char *end;
    double value = strtod(text, &end);
    if (*end || !isfinite(value) || (value == 0 && nonzero) ||
        (errno != 0 && errno != ERANGE)) return 0;
    *out = value;
    return 1;
}

static char *destination_key(const char *path, int *failed)
{
    char *copy = strdup(path);
    if (copy == NULL) { *failed = 1; return NULL; }
    char *slash = strrchr(copy, '/');
    const char *base = slash == NULL ? copy : slash + 1;
    char *parent;
    if (slash == NULL) parent = realpath(".", NULL);
    else {
        *slash = '\0';
        parent = realpath(slash == copy ? "/" : copy, NULL);
    }
    if (parent == NULL && errno == ENOMEM) *failed = 1;
    char *key = NULL;
    size_t size;
    if (parent != NULL && inkling_size_add(strlen(parent), strlen(base), &size) &&
        inkling_size_add(size, 2, &size)) {
        key = malloc(size);
        if (key == NULL) *failed = 1;
        if (key != NULL) (void)snprintf(key, size, "%s/%s", parent, base);
    }
    free(parent);
    free(copy);
    return key;
}

static int same_destination_names(char *a, char *b)
{
    char *left_base = strrchr(a, '/'), *right_base = strrchr(b, '/');
    *left_base++ = '\0';
    *right_base++ = '\0';
    struct stat left, right;
    if (stat(a, &left) != 0 || stat(b, &right) != 0) return -1;
    if (left.st_dev != right.st_dev || left.st_ino != right.st_ino) return 0;

    size_t bytes;
    if (!inkling_size_add(strlen(a), sizeof("/.inkling-destination-XXXXXX"), &bytes)) return -1;
    char *temporary = malloc(bytes);
    if (temporary == NULL) return -1;
    (void)snprintf(temporary, bytes, "%s/.inkling-destination-XXXXXX", a);
    if (mkdtemp(temporary) == NULL) { free(temporary); return -1; }
    int directory = open(temporary, O_RDONLY | O_DIRECTORY);
    int same = -1;
    if (directory >= 0) {
        int file = openat(directory, left_base, O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (file >= 0) {
            if (fstat(file, &left) == 0) {
                if (fstatat(directory, right_base, &right, 0) == 0)
                    same = left.st_dev == right.st_dev && left.st_ino == right.st_ino;
                else if (errno == ENOENT) same = 0;
            }
            if (close(file) != 0) same = -1;
            if (unlinkat(directory, left_base, 0) != 0) same = -1;
        }
        if (close(directory) != 0) same = -1;
    }
    if (rmdir(temporary) != 0) same = -1;
    free(temporary);
    return same;
}

static int same_destination(const char *a, const char *b)
{
    if (a == NULL || b == NULL) return 0;
    if (strcmp(a, b) == 0) return 1;
    struct stat left, right;
    if (stat(a, &left) == 0 && stat(b, &right) == 0 &&
        left.st_dev == right.st_dev && left.st_ino == right.st_ino) return 1;
    int failed = 0;
    char *left_key = destination_key(a, &failed), *right_key = destination_key(b, &failed);
    int same = 0;
    if (!failed && left_key != NULL && right_key != NULL)
        same = strcmp(left_key, right_key) == 0 ? 1 : same_destination_names(left_key, right_key);
    free(left_key);
    free(right_key);
    return failed ? -1 : same;
}

static int write_ids(FILE *file, const uint32_t *ids, size_t count)
{
    for (size_t i = 0; i < count; i++)
        if (fprintf(file, "%s%" PRIu32, i ? "," : "", ids[i]) < 0) return 0;
    return 1;
}

static int write_report(FILE *file, const uint32_t *prompt, size_t prompt_count,
                         const uint32_t *generated, const InklingGenerateResult *result)
{
    const char *reason = result->stop_reason == INKLING_STOP_EOS ? "eos" :
        result->stop_reason == INKLING_STOP_LENGTH ? "length" : "context";
    const InklingGenerateOptions *options = &result->effective;
    return fputs("{\"prompt_ids\":[", file) != EOF && write_ids(file, prompt, prompt_count) &&
        fputs("],\"generated_ids\":[", file) != EOF &&
        write_ids(file, generated, result->generated_count) &&
        fprintf(file, "],\"generated_count\":%zu,\"stop_reason\":\"%s\","
                "\"effective\":{\"max_new_tokens\":%zu,\"temperature\":%.17g,"
                "\"top_k\":%zu,\"top_p\":%.17g,\"seed\":\"%" PRIu64 "\"}}\n",
                result->generated_count, reason, options->max_new_tokens, options->temperature,
                options->top_k, options->top_p, options->seed) >= 0;
}

static int run_ids(int argc, char **argv)
{
    (void)signal(SIGPIPE, SIG_IGN);
    uint32_t *ids = NULL, *generated = NULL;
    float *logits = NULL;
    size_t tokens = 0, total = 0, count = 0;
    InklingGenerateOptions options = INKLING_GENERATE_OPTIONS_INIT;
    const char *logits_path = NULL;
    OutputFile token_file = {0}, report_file = {0};
    unsigned seen = 0;
    if (argc < 4 || !parse_ids(argv[3], &ids, &tokens)) goto invalid;
    for (int i = 4; i < argc; i += 2) {
        unsigned bit;
        uint64_t value;
        const char *flag = argv[i];
        if (i + 1 >= argc) goto invalid;
        const char *argument = argv[i + 1];
        if (strcmp(flag, "--gen") == 0) {
            bit = 1;
            if (!parse_unsigned(argument, SIZE_MAX, &value)) goto invalid;
            options.max_new_tokens = (size_t)value;
        } else if (strcmp(flag, "--temperature") == 0) {
            bit = 2;
            if (!parse_decimal(argument, &options.temperature) || options.temperature < 0) goto invalid;
        } else if (strcmp(flag, "--top-k") == 0) {
            bit = 4;
            if (!parse_unsigned(argument, SIZE_MAX, &value)) goto invalid;
            options.top_k = (size_t)value;
        } else if (strcmp(flag, "--top-p") == 0) {
            bit = 8;
            if (!parse_decimal(argument, &options.top_p) || options.top_p <= 0 || options.top_p > 1)
                goto invalid;
        } else if (strcmp(flag, "--seed") == 0) {
            bit = 16;
            if (!parse_unsigned(argument, UINT64_MAX, &options.seed)) goto invalid;
        } else if (strcmp(flag, "--dump-tokens") == 0) { bit = 32; token_file.path = argument; }
        else if (strcmp(flag, "--out") == 0) { bit = 64; report_file.path = argument; }
        else if (strcmp(flag, "--dump-logits") == 0) { bit = 128; logits_path = argument; }
        else goto invalid;
        if ((seen & bit) != 0 || *argument == '\0') goto invalid;
        seen |= bit;
    }
    if (seen != 128 && ((seen & 1) == 0 || (seen & 128) != 0)) goto invalid;
    if (!inkling_size_add(tokens, options.max_new_tokens, &total) ||
        same_destination(token_file.path, report_file.path)) goto invalid;
    InklingModel *model = NULL;
    InklingContext *context = NULL;
    int success = 0;
    if (!inkling_model_open(argv[1], &model)) goto done;
    if (seen == 128) {
        if (inkling_size_mul(tokens, inkling_model_logits_width(model), &count) &&
            inkling_alloc_array(count, sizeof(*logits), (void **)&logits) &&
            inkling_context_create(model, tokens, &context) &&
            inkling_forward(context, ids, tokens, logits, count) &&
            write_logits(logits_path, logits, count, tokens, inkling_model_logits_width(model))) success = 1;
        goto done;
    }
    size_t maximum = inkling_model_max_length(model);
    if (tokens > maximum) goto done;
    if (total > maximum) total = maximum;
    count = total - tokens;
    void *allocation = NULL;
    InklingGenerateResult result;
    if (!inkling_alloc_array(count, sizeof(*generated), &allocation)) goto done;
    generated = allocation;
    if (!inkling_context_create(model, total, &context) ||
        !inkling_generate(context, ids, tokens, &options, generated, count, &result)) goto done;
    if (token_file.path != NULL && (!stage_file(&token_file) ||
        !write_ids(token_file.file, generated, result.generated_count) ||
        fputc('\n', token_file.file) == EOF || !close_stage(&token_file))) goto done;
    if (report_file.path != NULL && (!stage_file(&report_file) ||
        !write_report(report_file.file, ids, tokens, generated, &result) ||
        !close_stage(&report_file))) goto done;
    if (token_file.path != NULL && !publish_file(&token_file)) goto done;
    if (report_file.path != NULL) {
        if (!publish_file(&report_file)) goto done;
    } else if (!write_report(stdout, ids, tokens, generated, &result) || fflush(stdout) != 0) goto done;
    success = 1;
done:
    if (!success) {
        fputs(seen == 128 ? "raw token forward failed\n" : "raw token generation or output failed\n", stderr);
        if (token_file.published) fprintf(stderr, "already published: %s\n", token_file.path);
        if (report_file.published) fprintf(stderr, "already published: %s\n", report_file.path);
    }
    discard_file(&token_file);
    discard_file(&report_file);
    free(ids);
    free(generated);
    free(logits);
    inkling_context_free(context);
    inkling_model_close(model);
    return success ? 0 : 1;
invalid:
    free(ids);
    fputs("invalid raw token options; expected --ids ID,ID,... with --gen N or --dump-logits PATH\n", stderr);
    return 2;
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
                "       %s MODEL_DIR --ids ID,ID,... --dump-logits PATH\n"
                "       %s MODEL_DIR --ids ID,ID,... --gen N [--temperature T] [--top-k K]\n"
                "         [--top-p P] [--seed SEED] [--dump-tokens PATH] [--out PATH]\n",
                argv[0], argv[0], argv[0], argv[0]);
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
