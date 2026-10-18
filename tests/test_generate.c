#define _POSIX_C_SOURCE 200809L
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inkling_model_internal.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); exit(1); } } while (0)

static size_t allocations_before_failure = SIZE_MAX;
void *inkling_test_malloc(size_t bytes)
{
    if (allocations_before_failure == 0) return NULL;
    if (allocations_before_failure != SIZE_MAX) allocations_before_failure--;
    return malloc(bytes);
}

typedef struct { size_t calls; const char *truncate_path; } Trace;
static void trace_step(void *user, const char *name, const float *values,
                        const size_t *indices, size_t count)
{
    Trace *trace = user;
    (void)values; (void)indices; (void)count;
    if (strcmp(name, "logits") == 0) {
        trace->calls++;
        if (trace->calls == 1 && trace->truncate_path != NULL) CHECK(truncate(trace->truncate_path, 0) == 0);
    }
}

static void unchanged(InklingContext *context, const uint32_t *prompt, size_t count,
                       const InklingGenerateOptions *options, size_t capacity)
{
    uint32_t out[16], old[16];
    InklingGenerateResult result, before;
    memset(out, 0xa5, sizeof(out)); memcpy(old, out, sizeof(out));
    memset(&result, 0x5a, sizeof(result)); memcpy(&before, &result, sizeof(result));
    CHECK(!inkling_generate(context, prompt, count, options, out, capacity, &result));
    CHECK(memcmp(out, old, sizeof(out)) == 0);
    CHECK(memcmp(&result, &before, sizeof(result)) == 0);
}

static void sample_guards(void)
{
    float logits[] = {1, 1, -1};
    InklingCandidate candidates[3];
    InklingGenerateOptions options = INKLING_GENERATE_OPTIONS_INIT, effective;
    uint64_t state = UINT64_MAX;
    uint32_t token = 99;
    CHECK(inkling_generate_options(&options, 3, &effective));
    CHECK(inkling_sample(logits, 3, &effective, candidates, &state, &token));
    CHECK(token == 0 && state == UINT64_MAX);
    logits[2] = NAN;
    CHECK(!inkling_sample(logits, 3, &effective, candidates, &state, &token));
    CHECK(token == 0 && state == UINT64_MAX);
    logits[2] = -INFINITY;
    CHECK(!inkling_sample(logits, 3, &effective, candidates, &state, &token));
    logits[0] = FLT_MAX; logits[1] = -FLT_MAX; logits[2] = 0;
    options.temperature = 0x1p-1074; options.top_k = SIZE_MAX;
    CHECK(inkling_generate_options(&options, 3, &effective));
    CHECK(effective.top_k == 3);
    CHECK(inkling_sample(logits, 3, &effective, candidates, &state, &token));
    CHECK(token == 0 && candidates[1].mass == 0 && candidates[2].mass == 0);
    CHECK(state == UINT64_MAX + UINT64_C(0x9e3779b97f4a7c15));
    logits[0] = 0; logits[1] = -745.1f;
    options.temperature = 1;
    CHECK(inkling_generate_options(&options, 2, &effective));
    CHECK(inkling_sample(logits, 2, &effective, candidates, &state, &token));
    CHECK(candidates[1].mass > 0);
}

static void read_failure(InklingModel *model, InklingContext *context)
{
    const char *original = model->shards[0].path;
    FILE *source = fopen(original, "rb");
    CHECK(source != NULL);
    char path[] = "/tmp/inkling-generate-read-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    FILE *copy = fdopen(fd, "wb");
    CHECK(copy != NULL);
    unsigned char buffer[4096];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), source)) != 0)
        CHECK(fwrite(buffer, 1, bytes, copy) == bytes);
    CHECK(!ferror(source)); CHECK(fclose(source) == 0); CHECK(fclose(copy) == 0);
    model->shards[0].path = path;
    Trace trace = {0, path};
    inkling_context_trace(context, trace_step, &trace);
    uint32_t prompt = 0;
    InklingGenerateOptions options = INKLING_GENERATE_OPTIONS_INIT;
    options.max_new_tokens = 4;
    unchanged(context, &prompt, 1, &options, 4);
    CHECK(trace.calls == 1);
    model->shards[0].path = (char *)original;
    inkling_context_trace(context, NULL, NULL);
    uint32_t out[4]; InklingGenerateResult result;
    CHECK(inkling_generate(context, &prompt, 1, &options, out, 4, &result));
    CHECK(result.generated_count == 4);
    CHECK(unlink(path) == 0);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    sample_guards();
    InklingModel *model = NULL;
    InklingContext *a = NULL, *b = NULL;
    CHECK(inkling_model_open_text_fixture(argv[1], &model));
    CHECK(inkling_context_create(model, 16, &a));
    CHECK(inkling_context_create(model, 16, &b));
    size_t width = inkling_model_logits_width(model);
    CHECK(inkling_model_max_length(model) == model->config.model_max_length);
    CHECK(inkling_model_max_length(NULL) == 0);
    float full[16 * 40], last[40]; uint32_t prefix[] = {2, 5, 7, 39, 8, 9};
    for (size_t n = 1; n <= 6; n++) {
        CHECK(inkling_forward(a, prefix, n, full, 16 * 40));
        CHECK(inkling_forward_last(a, prefix, n, last, 40));
        CHECK(memcmp(full + (n - 1) * width, last, width * sizeof(float)) == 0);
    }
    InklingGenerateOptions options = INKLING_GENERATE_OPTIONS_INIT;
    InklingGenerateResult result, repeat;
    uint32_t prompt = 2, out[16], again[16];
    Trace trace = {0, NULL};
    inkling_context_trace(a, trace_step, &trace);
    allocations_before_failure = 0;
    CHECK(inkling_generate(a, &prompt, 1, &options, NULL, 0, &result));
    CHECK(a->generation == NULL && trace.calls == 0 && result.stop_reason == INKLING_STOP_LENGTH);
    options.max_new_tokens = 6;
    for (size_t i = 0; i < 4; i++) {
        allocations_before_failure = i;
        unchanged(a, &prompt, 1, &options, 6);
        CHECK(a->generation == NULL && trace.calls == 0);
    }
    allocations_before_failure = SIZE_MAX;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 6, &result));
    CHECK(result.generated_count == 1 && out[0] == 6 && result.stop_reason == INKLING_STOP_EOS);
    prompt = 1;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 6, &result));
    CHECK(result.generated_count == 3 && out[0] == 14 && out[1] == 2 && out[2] == 6);
    prompt = 39;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 6, &result));
    CHECK(result.generated_count == 3 && out[0] == 14 && out[1] == 2 && out[2] == 6);
    prompt = 6;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 6, &result));
    CHECK(result.generated_count > 0);
    options.temperature = 0.7; options.top_k = 9; options.top_p = 0.8; options.seed = UINT64_MAX;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 6, &result));
    CHECK(inkling_generate(b, &prompt, 1, &options, again, 6, &repeat));
    CHECK(result.generated_count == repeat.generated_count && memcmp(out, again, result.generated_count * 4) == 0);
    CHECK(inkling_forward(a, prefix, 6, full, 16 * 40));
    inkling_context_reset(a);
    CHECK(inkling_generate(a, &prompt, 1, &options, again, 6, &repeat));
    CHECK(result.generated_count == repeat.generated_count && memcmp(out, again, result.generated_count * 4) == 0);
    options.temperature = NAN; unchanged(a, &prompt, 1, &options, 6);
    options.temperature = INFINITY; unchanged(a, &prompt, 1, &options, 6);
    options.temperature = -1; unchanged(a, &prompt, 1, &options, 6);
    options.temperature = 0; options.top_p = NAN; unchanged(a, &prompt, 1, &options, 6);
    options.top_p = 0; unchanged(a, &prompt, 1, &options, 6);
    options.top_p = 1.1; unchanged(a, &prompt, 1, &options, 6);
    options.top_p = 1; options.max_new_tokens = SIZE_MAX; unchanged(a, &prompt, 1, &options, 6);
    options.max_new_tokens = 6; unchanged(a, &prompt, 0, &options, 6);
    unchanged(a, NULL, 1, &options, 6); unchanged(a, &prompt, 1, &options, 5);
    unchanged(a, &prompt, SIZE_MAX, &options, 6);
    unchanged(a, &prompt, 1, &options, SIZE_MAX);
    unchanged(a, (const uint32_t *)(UINTPTR_MAX - 1), 1, &options, 6);
    prompt = 40; unchanged(a, &prompt, 1, &options, 6); prompt = 0;
    CHECK(!inkling_generate(a, &prompt, 1, &options, NULL, 6, &result));
    CHECK(!inkling_generate(a, &prompt, 1, &options, out, 6, NULL));
    CHECK(!inkling_generate(a, &prompt, 1, NULL, out, 6, &result));
    CHECK(!inkling_generate(a, out, 1, &options, out, 6, &result));
    CHECK(!inkling_generate(a, &prompt, 1, &options, (uint32_t *)&options, 6, &result));
    CHECK(!inkling_generate(a, &prompt, 1, &options, (uint32_t *)&result, 6, &result));
    CHECK(!inkling_generate(a, &prompt, 1, &options, out, 6, (InklingGenerateResult *)&options));
    InklingContext *small = NULL;
    CHECK(inkling_context_create(model, 1, &small));
    CHECK(inkling_generate(small, &prompt, 1, &options, NULL, 0, &result));
    CHECK(result.generated_count == 0 && result.stop_reason == INKLING_STOP_CONTEXT && small->generation == NULL);
    inkling_context_free(small);
    CHECK(inkling_context_create(model, 2, &small));
    options.max_new_tokens = 1;
    CHECK(inkling_generate(small, &prompt, 1, &options, out, 1, &result));
    CHECK(result.stop_reason == INKLING_STOP_LENGTH);
    options.max_new_tokens = 5;
    CHECK(inkling_generate(small, &prompt, 1, &options, out, 1, &result));
    CHECK(result.stop_reason == INKLING_STOP_CONTEXT);
    prompt = 2; options.max_new_tokens = 1;
    CHECK(inkling_generate(small, &prompt, 1, &options, out, 1, &result));
    CHECK(result.stop_reason == INKLING_STOP_EOS);
    inkling_context_free(small);
    uint32_t saved_eos = model->config.eos_token_id;
    model->config.eos_token_id = 39;
    prompt = 2; options.max_new_tokens = 1;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 1, &result));
    CHECK(out[0] == 6 && result.stop_reason == INKLING_STOP_LENGTH);
    model->config.eos_token_id = saved_eos;
    model->complete = 0;
    unchanged(a, &prompt, 1, &options, 1);
    CHECK(inkling_model_max_length(model) == 0);
    model->complete = 1;
    uint32_t saved_limit = model->config.model_max_length;
    model->config.model_max_length = 2;
    prompt = 0; options.max_new_tokens = 5;
    CHECK(inkling_generate(a, &prompt, 1, &options, out, 1, &result));
    CHECK(result.generated_count == 1 && result.stop_reason == INKLING_STOP_CONTEXT);
    unchanged(a, prefix, 3, &options, 1);
    model->config.model_max_length = saved_limit;
    struct { uint32_t before, ids[6], after; } guarded = {123, {0}, 456};
    prompt = 1; options.max_new_tokens = 6;
    CHECK(inkling_generate(a, &prompt, 1, &options, guarded.ids, 6, &result));
    CHECK(guarded.before == 123 && guarded.after == 456);
    CHECK(guarded.ids[3] == 0 && guarded.ids[4] == 0 && guarded.ids[5] == 0);
    read_failure(model, a);
    inkling_context_free(a); inkling_context_free(b); inkling_model_close(model);
    puts("generation guards, replay, parity and failure checks passed");
    return 0;
}
