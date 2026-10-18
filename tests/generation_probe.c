#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling_model_internal.h"
#include "inkling_alloc.h"
#include "inkling_json.h"

typedef struct { float *logits; size_t width, capacity, count; int failed; } Trace;

static void capture(void *user, const char *name, const float *values,
                     const size_t *indices, size_t count)
{
    Trace *trace = user;
    (void)indices;
    if (strcmp(name, "logits") != 0) return;
    if (count != trace->width || trace->count >= trace->capacity) { trace->failed = 1; return; }
    memcpy(trace->logits + trace->count * trace->width, values, count * sizeof(float));
    trace->count++;
}

static int size_field(const InklingJsonValue *object, const char *key, size_t *out)
{
    uint64_t value;
    if (!inkling_json_number_u64(inkling_json_object_get(object, key), &value) || value > SIZE_MAX)
        return 0;
    *out = (size_t)value;
    return 1;
}

static int read_double(const InklingJsonValue *value, double *out)
{
    size_t length;
    const char *text = inkling_json_number_text(value, &length);
    if (text == NULL) return 0;
    char *end;
    double number = strtod(text, &end);
    if (end != text + length || !isfinite(number)) return 0;
    *out = number;
    return 1;
}

static int read_options(const InklingJsonValue *test, InklingGenerateOptions *options)
{
    const InklingJsonValue *object = inkling_json_object_get(test, "options");
    return size_field(object, "max_new_tokens", &options->max_new_tokens) &&
        size_field(object, "top_k", &options->top_k) &&
        read_double(inkling_json_object_get(object, "temperature"), &options->temperature) &&
        read_double(inkling_json_object_get(object, "top_p"), &options->top_p) &&
        inkling_json_number_u64(inkling_json_object_get(object, "seed"), &options->seed);
}

static int report_options(FILE *file, const InklingGenerateOptions *options)
{
    return fprintf(file, "\"effective\":{\"max_new_tokens\":%zu,\"temperature\":%.17g,"
                   "\"top_k\":%zu,\"top_p\":%.17g,\"seed\":\"%" PRIu64 "\"}",
                   options->max_new_tokens, options->temperature, options->top_k,
                   options->top_p, options->seed) >= 0;
}

int main(int argc, char **argv)
{
    if (argc != 5) return 2;
    int sampler = strcmp(argv[1], "--sampler") == 0;
    char *end;
    errno = 0;
    unsigned long long number = strtoull(argv[3], &end, 10);
    if (errno || !*argv[3] || argv[3][0] == '-' || *end || number > SIZE_MAX) return 2;
    InklingJsonDocument document = {0};
    InklingModel *model = NULL;
    InklingContext *context = NULL;
    uint32_t *prompt = NULL, *generated = NULL;
    float *logits = NULL;
    InklingCandidate *candidates = NULL;
    FILE *file = NULL;
    int success = 0;
    void *allocation = NULL;
    if (!inkling_json_load_file(argv[2], 64L * 1024L * 1024L, &document)) goto done;
    const InklingJsonValue *test = inkling_json_array_at(inkling_json_object_get(document.root,
        sampler ? "sampler" : "generation"), (size_t)number);
    InklingGenerateOptions options, effective;
    if (test == NULL || !read_options(test, &options)) goto done;
    if (sampler) {
        const InklingJsonValue *array = inkling_json_object_get(test, "logits");
        size_t width = inkling_json_array_size(array);
        if (!inkling_alloc_array(width, sizeof(float), &allocation)) goto done;
        logits = allocation;
        if (!inkling_alloc_array(width, sizeof(InklingCandidate), &allocation)) goto done;
        candidates = allocation;
        for (size_t i = 0; i < width; i++) {
            double value;
            if (!read_double(inkling_json_array_at(array, i), &value) ||
                !isfinite(value) || !isfinite((float)value)) goto done;
            logits[i] = (float)value;
        }
        uint64_t state = options.seed;
        uint32_t token;
        if (!inkling_generate_options(&options, width, &effective) ||
            !inkling_sample(logits, width, &effective, candidates, &state, &token)) goto done;
        file = fopen(argv[4], "w");
        if (file == NULL || fprintf(file, "{\"token_id\":%" PRIu32 ",\"state_hex\":\"%016" PRIx64 "\",",
                                    token, state) < 0 || !report_options(file, &effective)) goto done;
        size_t retained = effective.top_k;
        double mass_total = 0;
        if (effective.temperature == 0) {
            candidates[0].id = token;
            candidates[0].mass = 1;
        }
        for (size_t i = 0; i < retained; i++) mass_total += candidates[i].mass;
        if (effective.top_p < 1) {
            double cumulative = 0;
            for (size_t i = 0; i < retained; i++) {
                cumulative += candidates[i].mass;
                if (cumulative >= effective.top_p * mass_total) {
                    retained = i + 1;
                    break;
                }
            }
        }
        if (fputs(",\"support_ids\":[", file) == EOF) goto done;
        for (size_t i = 0; i < retained; i++)
            if (fprintf(file, "%s%" PRIu32, i ? "," : "", candidates[i].id) < 0) goto done;
        if (fputs("],\"masses\":[", file) == EOF) goto done;
        for (size_t i = 0; i < retained; i++)
            if (fprintf(file, "%s%.17g", i ? "," : "", candidates[i].mass) < 0) goto done;
        if (fputs("]}\n", file) == EOF) goto done;
    } else {
        const InklingJsonValue *array = inkling_json_object_get(test, "prompt_ids");
        size_t prompt_count = inkling_json_array_size(array), capacity, trace_count;
        if (!size_field(test, "context_capacity", &capacity) || prompt_count == 0 || prompt_count > capacity ||
            !inkling_alloc_array(prompt_count, sizeof(uint32_t), &allocation)) goto done;
        prompt = allocation;
        for (size_t i = 0; i < prompt_count; i++) {
            uint64_t value;
            if (!inkling_json_number_u64(inkling_json_array_at(array, i), &value) || value > UINT32_MAX)
                goto done;
            prompt[i] = (uint32_t)value;
        }
        if (!inkling_model_open_text_fixture(argv[1], &model) ||
            !inkling_context_create(model, capacity, &context)) goto done;
        size_t count = capacity - prompt_count, width = inkling_model_logits_width(model);
        if (count > options.max_new_tokens) count = options.max_new_tokens;
        if (!inkling_alloc_array(count, sizeof(uint32_t), &allocation)) goto done;
        generated = allocation;
        if (!inkling_size_mul(count, width, &trace_count) ||
            !inkling_alloc_array(trace_count, sizeof(float), &allocation)) goto done;
        logits = allocation;
        Trace trace = {logits, width, count, 0, 0};
        inkling_context_trace(context, capture, &trace);
        InklingGenerateResult result;
        if (!inkling_generate(context, prompt, prompt_count, &options, generated, count, &result) ||
            trace.failed || trace.count != result.generated_count) goto done;
        const char *reason = result.stop_reason == INKLING_STOP_EOS ? "eos" :
            result.stop_reason == INKLING_STOP_LENGTH ? "length" : "context";
        file = fopen(argv[4], "w");
        if (file == NULL || fputs("{\"generated_ids\":[", file) == EOF) goto done;
        for (size_t i = 0; i < result.generated_count; i++)
            if (fprintf(file, "%s%" PRIu32, i ? "," : "", generated[i]) < 0) goto done;
        if (fprintf(file, "],\"generated_count\":%zu,\"stop_reason\":\"%s\",", result.generated_count, reason) < 0 ||
            !report_options(file, &result.effective) || fputs(",\"steps\":[", file) == EOF) goto done;
        for (size_t i = 0; i < trace.count; i++) {
            if (fprintf(file, "%s{\"token_id\":%" PRIu32 ",\"logits\":[", i ? "," : "", generated[i]) < 0) goto done;
            for (size_t j = 0; j < width; j++)
                if (fprintf(file, "%s%.9g", j ? "," : "", (double)logits[i * width + j]) < 0) goto done;
            if (fputs("]}", file) == EOF) goto done;
        }
        if (fputs("]}\n", file) == EOF) goto done;
    }
    success = 1;
done:
    if (file != NULL && fclose(file) != 0) success = 0;
    free(prompt); free(generated); free(logits); free(candidates);
    inkling_context_free(context);
    inkling_model_close(model);
    inkling_json_document_free(&document);
    if (!success) fputs("generation probe failed\n", stderr);
    return success ? 0 : 1;
}
