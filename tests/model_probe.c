#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling_model_internal.h"
#include "../src/core/inkling_alloc.h"
#include "../src/io/inkling_json.h"

typedef struct {
    FILE *file;
    const InklingJsonValue *expected;
    int first, failed;
    size_t sample_offsets[2];
} TraceOutput;

static void trace_output(void *user, const char *name, const float *values,
                         const size_t *indices, size_t count)
{
    TraceOutput *trace = user;
    const InklingJsonValue *wanted = inkling_json_object_get(
        inkling_json_object_get(trace->expected, indices ? "indices" : "traces"), name);
    if (wanted == NULL) return;
    const InklingJsonValue *samples = inkling_json_object_get(
        inkling_json_object_get(trace->expected, "trace_samples"), name);
    size_t offset = 0, sampled = 0;
    if (samples != NULL) {
        size_t slot;
        if (strcmp(name, "attention_probs") == 0) slot = 0;
        else if (strcmp(name, "relative_bias") == 0) slot = 1;
        else { trace->failed = 1; return; }
        offset = trace->sample_offsets[slot];
        if (indices != NULL || !inkling_size_add(offset, count, &trace->sample_offsets[slot])) {
            trace->failed = 1;
            return;
        }
        for (size_t i = 0; i < inkling_json_array_size(samples); i++) {
            uint64_t position;
            if (!inkling_json_number_u64(inkling_json_array_at(samples, i), &position) || position > SIZE_MAX) {
                trace->failed = 1;
                return;
            }
            if (position >= offset && position - offset < count) sampled++;
        }
        if (sampled == 0) return;
    }
    if (fprintf(trace->file, "%s{\"name\":\"%s\",\"%s\":[",
                trace->first ? "" : ",", name, indices ? "indices" : "values") < 0)
        trace->failed = 1;
    trace->first = 0;
    size_t written_count = 0;
    size_t length = samples ? inkling_json_array_size(samples) : count;
    for (size_t item = 0; item < length; item++) {
        size_t i = item;
        if (samples != NULL) {
            uint64_t position;
            if (!inkling_json_number_u64(inkling_json_array_at(samples, item), &position)) {
                trace->failed = 1;
                return;
            }
            if (position < offset || position - offset >= count) continue;
            i = (size_t)position - offset;
        }
        int written;
        if (indices != NULL) {
            written = fprintf(trace->file, "%s%zu", written_count ? "," : "", indices[i]);
        } else {
            if (!isfinite(values[i])) trace->failed = 1;
            written = fprintf(trace->file, "%s%.9g", written_count ? "," : "", (double)values[i]);
        }
        written_count++;
        if (written < 0) trace->failed = 1;
    }
    if (fputs("]}", trace->file) == EOF) trace->failed = 1;
}

static int size_field(const InklingJsonValue *object, const char *name, size_t *out)
{
    uint64_t value;
    if (!inkling_json_number_u64(inkling_json_object_get(object, name), &value) ||
        value > SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int run_case(InklingContext *context, const InklingModel *model, const InklingLayer *layer,
                    const char *kind, const uint32_t *ids, const float *input,
                    size_t tokens, size_t position, float *out, size_t count, size_t *selected)
{
    if (tokens == 0) return 0;
    if (strcmp(kind, "router") == 0)
        return tokens == 1 && inkling_router_forward(model, layer, input, selected, out);
    if (strcmp(kind, "model") == 0)
        return inkling_forward(context, ids, tokens, out, count);
    if (strcmp(kind, "layer") == 0)
        return inkling_layer_forward(context, layer, input, tokens, out);
    if (strcmp(kind, "attention") == 0)
        return inkling_attention_forward(context, layer, input, tokens, position, out);
    if (strcmp(kind, "mlp") == 0)
        return inkling_mlp_forward(context, layer, input, tokens, out);
    return 0;
}

static int numbers(const InklingJsonValue *array, size_t count, float *out)
{
    if (inkling_json_array_size(array) != count) return 0;
    for (size_t i = 0; i < count; i++) {
        double value;
        if (!inkling_json_number_double(inkling_json_array_at(array, i), &value) ||
            !isfinite(value) || !isfinite((float)value)) return 0;
        out[i] = (float)value;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc == 4 && strcmp(argv[1], "--bind-layer") == 0) {
        InklingModel *model = NULL;
        int success = inkling_model_open_layer_fixture(argv[2], argv[3], &model);
        inkling_model_close(model);
        return success ? 0 : 1;
    }
    if (argc != 5 && argc != 6) {
        fputs("usage: model_probe FIXTURE_DIR CASES_JSON CASE_INDEX OUTPUT_JSON [RANGE_MANIFEST]\n", stderr);
        return 2;
    }
    char *end = NULL;
    errno = 0;
    unsigned long long index = strtoull(argv[3], &end, 10);
    if (errno || !*argv[3] || argv[3][0] == '-' || *end || index > SIZE_MAX) return 2;
    InklingJsonDocument document = {0};
    InklingModel *model = NULL;
    InklingContext *a = NULL, *b = NULL;
    FILE *file = NULL;
    float *input = NULL, *output = NULL, *repeat = NULL;
    float *bias = NULL;
    size_t *selected = NULL;
    uint32_t *ids = NULL;
    int success = 0;
    if (!inkling_json_load_file(argv[2], 64L * 1024L * 1024L, &document)) goto done;
    const InklingJsonValue *test = inkling_json_array_at(
        inkling_json_object_get(document.root, "cases"), (size_t)index);
    const char *kind = inkling_json_string(inkling_json_object_get(test, "kind"), NULL);
    size_t tokens, position = 0, input_count, output_count;
    if (test == NULL || kind == NULL || !size_field(test, "tokens", &tokens) || !tokens) goto done;
    if (inkling_json_object_get(test, "position_start") != NULL &&
        !size_field(test, "position_start", &position)) goto done;
    if (!(argc == 6 ? inkling_model_open_layer_fixture(argv[1], argv[5], &model)
                     : inkling_model_open_text_fixture(argv[1], &model))) goto done;
    int router = strcmp(kind, "router") == 0;
    if (router) {
        if (!inkling_size_add(model->config.num_routed_experts, model->config.num_shared_experts,
                             &input_count) ||
            !inkling_size_add(model->config.num_experts_per_token, model->config.num_shared_experts,
                             &output_count)) goto done;
    } else if (!inkling_size_mul(tokens, model->config.hidden_size, &input_count) ||
        !inkling_size_mul(tokens, strcmp(kind, "model") == 0
                         ? inkling_model_logits_width(model) : model->config.hidden_size,
                         &output_count)) goto done;
    if (!inkling_alloc_array(input_count, sizeof(*input), (void **)&input) ||
        !inkling_alloc_array(tokens, sizeof(*ids), (void **)&ids) ||
        !inkling_alloc_array(output_count, sizeof(*output), (void **)&output) ||
        !inkling_alloc_array(output_count, sizeof(*repeat), (void **)&repeat)) goto done;
    const InklingLayer *layer = NULL;
    if (strcmp(kind, "model") == 0) {
        const InklingJsonValue *array = inkling_json_object_get(test, "input_ids");
        if (inkling_json_array_size(array) != tokens) goto done;
        for (size_t i = 0; i < tokens; i++) {
            uint64_t value;
            if (!inkling_json_number_u64(inkling_json_array_at(array, i), &value) ||
                value > UINT32_MAX) goto done;
            ids[i] = (uint32_t)value;
        }
    } else {
        size_t original_id;
        const InklingJsonValue *array = inkling_json_object_get(test, "input");
        if (!size_field(test, "layer", &original_id)) goto done;
        for (size_t i = 0; i < model->layer_count; i++)
            if (model->layers[i].original_id == original_id) layer = &model->layers[i];
        if (layer == NULL) goto done;
        if (!numbers(array, input_count, input)) goto done;
    }
    InklingModel router_model;
    InklingLayer router_layer;
    const InklingModel *run_model = model;
    if (router) {
        double route_scale, global_scale;
        if (!inkling_alloc_array(model->config.num_routed_experts, sizeof(*bias), (void **)&bias) ||
            !inkling_alloc_array(model->config.num_experts_per_token, sizeof(*selected), (void **)&selected) ||
            !numbers(inkling_json_object_get(test, "bias"), model->config.num_routed_experts, bias) ||
            !inkling_json_number_double(inkling_json_object_get(test, "route_scale"), &route_scale) ||
            !inkling_json_number_double(inkling_json_object_get(test, "global_scale"), &global_scale) ||
            !isfinite((float)route_scale) || !isfinite((float)global_scale)) goto done;
        router_model = *model;
        router_layer = *layer;
        router_model.config.route_scale = (float)route_scale;
        router_layer.global_scale = (float)global_scale;
        router_layer.correction = bias;
        run_model = &router_model;
        layer = &router_layer;
    }
    if (!inkling_context_create(model, tokens, &a) ||
        !inkling_context_create(model, tokens, &b)) goto done;
    file = fopen(argv[4], "w");
    if (file == NULL || fputs("{\"events\":[", file) == EOF) goto done;
    TraceOutput trace = {file, test, 1, 0, {0, 0}};
    inkling_context_trace(a, trace_output, &trace);
    if (!run_case(a, run_model, layer, kind, ids, input, tokens, position, output, output_count, selected) ||
        trace.failed) goto done;
    if (router) trace_output(&trace, "router_selected", NULL, selected, model->config.num_experts_per_token);
    inkling_context_trace(a, NULL, NULL);
    if (!run_case(b, run_model, layer, kind, ids, input, tokens, position, repeat, output_count, selected) ||
        memcmp(output, repeat, output_count * sizeof(*output)) != 0) goto done;
    inkling_context_reset(a);
    if (!run_case(a, run_model, layer, kind, ids, input, tokens, position, repeat, output_count, selected) ||
        memcmp(output, repeat, output_count * sizeof(*output)) != 0) goto done;
    if (run_case(a, run_model, layer, kind, ids, input, 0, position, repeat, output_count, selected)) goto done;
    if (!run_case(a, run_model, layer, kind, ids, input, tokens, position, repeat, output_count, selected) ||
        memcmp(output, repeat, output_count * sizeof(*output)) != 0) goto done;
    if (fputs("],\"output\":[", file) == EOF) goto done;
    for (size_t i = 0; i < output_count; i++) {
        if (!isfinite(output[i]) || fprintf(file, "%s%.9g", i ? "," : "", (double)output[i]) < 0)
            goto done;
    }
    if (fputs("]}\n", file) == EOF) goto done;
    success = 1;
done:
    if (file != NULL && fclose(file) != 0) success = 0;
    if (!success) fputs("model probe failed\n", stderr);
    free(input);
    free(output);
    free(repeat);
    free(ids);
    free(bias);
    free(selected);
    inkling_context_free(a);
    inkling_context_free(b);
    inkling_model_close(model);
    inkling_json_document_free(&document);
    return success ? 0 : 1;
}
