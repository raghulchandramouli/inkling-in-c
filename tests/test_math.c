#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/inkling_math.h"
#include "../src/io/inkling_json.h"

#define CAPACITY 4096
#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

typedef const InklingJsonValue Json;

static Json *field(Json *value, const char *name)
{
    Json *result = inkling_json_object_get(value, name);
    CHECK(result != NULL);
    return result;
}

static uint64_t integer(Json *value)
{
    uint64_t result;
    CHECK(inkling_json_number_u64(value, &result));
    return result;
}

static size_t dimension(Json *value, const char *name)
{
    uint64_t result = integer(field(value, name));
    CHECK(result > 0 && result <= CAPACITY);
    return (size_t)result;
}

static double number(Json *value)
{
    double result;
    CHECK(inkling_json_number_double(value, &result));
    return result;
}

static size_t floats(Json *record, const char *key, float *out)
{
    Json *values = field(record, key);
    size_t count = inkling_json_array_size(values);
    CHECK(count > 0 && count <= CAPACITY);
    for (size_t i = 0; i < count; i++) {
        uint64_t encoded = integer(inkling_json_array_at(values, i));
        CHECK(encoded <= UINT32_MAX);
        uint32_t bits = (uint32_t)encoded;
        memcpy(&out[i], &bits, sizeof(bits));
    }
    return count;
}

static size_t bytes(Json *record, const char *key, uint8_t *out)
{
    Json *values = field(record, key);
    size_t count = inkling_json_array_size(values);
    CHECK(count > 0 && count <= CAPACITY);
    for (size_t i = 0; i < count; i++) {
        uint64_t value = integer(inkling_json_array_at(values, i));
        CHECK(value <= UINT8_MAX);
        out[i] = (uint8_t)value;
    }
    return count;
}

static void close_values(Json *root, const char *operation, const float *actual,
                         const float *expected, size_t count)
{
    Json *tolerance = field(field(root, "tolerances"), operation);
    double atol = number(field(tolerance, "atol"));
    double rtol = number(field(tolerance, "rtol"));
    for (size_t i = 0; i < count; i++) {
        double error = fabs((double)actual[i] - expected[i]);
        if (!isfinite(actual[i]) || !isfinite(expected[i]) || error > atol + rtol * fabs(expected[i])) {
            fprintf(stderr, "%s[%zu]: got %.9g, expected %.9g (error %.9g)\n",
                    operation, i, (double)actual[i], (double)expected[i], error);
            exit(1);
        }
    }
}

static void bf16_cases(Json *root)
{
    Json *cases = field(root, "bf16");
    CHECK(inkling_json_array_size(cases) == 7);
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t rows = dimension(item, "rows"), cols = dimension(item, "columns");
        uint16_t weights[CAPACITY];
        float x[CAPACITY], out[CAPACITY], expected[CAPACITY];
        CHECK(rows * cols <= CAPACITY);
        Json *values = field(item, "weights");
        CHECK(inkling_json_array_size(values) == rows * cols);
        for (size_t i = 0; i < rows * cols; i++) {
            uint64_t value = integer(inkling_json_array_at(values, i));
            CHECK(value <= UINT16_MAX);
            weights[i] = (uint16_t)value;
        }
        CHECK(floats(item, "x", x) == cols && floats(item, "expected", expected) == rows);
        out[rows] = 123;
        CHECK(inkling_bf16_matvec(weights, rows * cols, rows, cols, x, cols, out, rows));
        CHECK(out[rows] == 123);
        close_values(root, "matvec", out, expected, rows);
        const char *name = inkling_json_string(field(item, "name"), NULL);
        CHECK(name != NULL);
        if (strcmp(name, "bf16_boundary") == 0)
            CHECK(memcmp(out, expected, rows * sizeof(float)) == 0);
    }
}

static void nvfp4_cases(Json *root)
{
    Json *cases = field(root, "nvfp4");
    CHECK(inkling_json_array_size(cases) == 7);
    size_t nibble_changes = 0, scale_changes = 0, expert_changes = 0;
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t experts = dimension(item, "experts"), rows = dimension(item, "rows");
        size_t cols = dimension(item, "columns");
        uint64_t shape[3] = {experts, rows, cols};
        uint64_t packed_shape[3] = {experts, rows, cols / 2};
        uint64_t scales_shape[3] = {experts, rows, cols / 16};
        InklingNvfp4Layout layout;
        CHECK(inkling_nvfp4_layout(shape, packed_shape, scales_shape, experts, &layout));
        uint8_t packed[CAPACITY], scales[CAPACITY], changed[CAPACITY];
        float globals[CAPACITY], x[CAPACITY], out[CAPACITY], expected[CAPACITY], wrong[CAPACITY];
        CHECK(bytes(item, "packed", packed) == layout.packed_bytes);
        CHECK(bytes(item, "scales", scales) == layout.scale_bytes);
        CHECK(floats(item, "scale2", globals) == experts);
        CHECK(floats(item, "x", x) == cols);
        CHECK(floats(item, "expected", expected) == experts * rows);
        for (size_t e = 0; e < experts; e++) {
            out[rows] = 123;
            CHECK(inkling_nvfp4_matvec(&layout, e, packed, layout.packed_bytes,
                    scales, layout.scale_bytes, globals, experts, x, cols, out, rows));
            CHECK(out[rows] == 123);
            close_values(root, "matvec", out, expected + e * rows, rows);
            for (size_t i = 0; i < layout.packed_bytes; i++)
                changed[i] = (uint8_t)((packed[i] >> 4) | ((unsigned int)packed[i] << 4));
            CHECK(inkling_nvfp4_matvec(&layout, e, changed, layout.packed_bytes,
                    scales, layout.scale_bytes, globals, experts, x, cols, wrong, rows));
            for (size_t i = 0; i < rows; i++) nibble_changes += fabsf(out[i] - wrong[i]) > 0.001f ? 1U : 0U;
            for (size_t i = 0; i < layout.scale_bytes; i++)
                changed[i] = scales[(i + 1) % layout.scale_bytes];
            CHECK(inkling_nvfp4_matvec(&layout, e, packed, layout.packed_bytes,
                    changed, layout.scale_bytes, globals, experts, x, cols, wrong, rows));
            for (size_t i = 0; i < rows; i++) scale_changes += fabsf(out[i] - wrong[i]) > 0.001f ? 1U : 0U;
            if (e > 0) {
                float saved = globals[e];
                globals[e] = globals[0];
                CHECK(inkling_nvfp4_matvec(&layout, e, packed, layout.packed_bytes,
                        scales, layout.scale_bytes, globals, experts, x, cols, wrong, rows));
                globals[e] = saved;
                for (size_t i = 0; i < rows; i++) expert_changes += fabsf(out[i] - wrong[i]) > 0.001f ? 1U : 0U;
            }
        }
    }
    CHECK(nibble_changes > 0 && scale_changes > 0 && expert_changes > 0);
}

static void vector_cases(Json *root)
{
    float x[CAPACITY], weights[CAPACITY], out[CAPACITY], expected[CAPACITY], inplace[CAPACITY];
    Json *cases = field(root, "norm");
    CHECK(inkling_json_array_size(cases) == 7);
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t n = floats(item, "x", x);
        CHECK(floats(item, "weights", weights) == n && floats(item, "expected", expected) == n);
        float epsilon = (float)number(field(item, "epsilon"));
        CHECK(inkling_rmsnorm(x, n, weights, n, epsilon, out, n));
        close_values(root, "norm", out, expected, n);
        memcpy(inplace, x, n * sizeof(float));
        CHECK(inkling_rmsnorm(inplace, n, weights, n, epsilon, inplace, n));
        CHECK(memcmp(inplace, out, n * sizeof(float)) == 0);
    }
    const char *activation_names[] = {"sigmoid", "silu"};
    int (*activations[])(const float *, size_t, float *, size_t) = {inkling_sigmoid, inkling_silu};
    cases = field(root, "activation");
    CHECK(inkling_json_array_size(cases) == 3);
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t n = floats(item, "x", x);
        for (size_t a = 0; a < 2; a++) {
            CHECK(floats(item, activation_names[a], expected) == n);
            CHECK(activations[a](x, n, out, n));
            close_values(root, "activation", out, expected, n);
            memcpy(inplace, x, n * sizeof(float));
            CHECK(activations[a](inplace, n, inplace, n));
            CHECK(memcmp(inplace, out, n * sizeof(float)) == 0);
        }
    }
    cases = field(root, "softmax");
    CHECK(inkling_json_array_size(cases) == 6);
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t n = floats(item, "x", x);
        CHECK(floats(item, "expected", expected) == n);
        CHECK(inkling_softmax(x, n, out, n));
        close_values(root, "softmax", out, expected, n);
        double sum = 0;
        for (size_t i = 0; i < n; i++) {
            if (x[i] == -INFINITY) CHECK(out[i] == 0.0f);
            CHECK(out[i] >= 0 && out[i] <= 1);
            sum += out[i];
        }
        CHECK(fabs(sum - 1) < 2e-7);
        memcpy(inplace, x, n * sizeof(float));
        CHECK(inkling_softmax(inplace, n, inplace, n));
        for (size_t i = 0; i < n; i++)
            if (x[i] == -INFINITY) CHECK(inplace[i] == 0.0f);
        CHECK(memcmp(inplace, out, n * sizeof(float)) == 0);
    }
    cases = field(root, "topk");
    CHECK(inkling_json_array_size(cases) == 5);
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t n = floats(item, "x", x), k = dimension(item, "k"), indices[CAPACITY];
        indices[k] = SIZE_MAX;
        CHECK(inkling_topk(x, n, k, indices, k));
        CHECK(indices[k] == SIZE_MAX);
        Json *values = field(item, "expected");
        CHECK(inkling_json_array_size(values) == k);
        for (size_t i = 0; i < k; i++) CHECK(indices[i] == integer(inkling_json_array_at(values, i)));
    }
}

static void conv_cases(Json *root)
{
    Json *cases = field(root, "conv");
    CHECK(inkling_json_array_size(cases) == 8);
    for (size_t c = 0; c < inkling_json_array_size(cases); c++) {
        Json *item = inkling_json_array_at(cases, c);
        size_t tokens = dimension(item, "tokens"), channels = dimension(item, "channels");
        size_t n = tokens * channels, history = channels * 3, kernel = channels * 4;
        float x[CAPACITY], weights[CAPACITY], initial[CAPACITY], state[CAPACITY], final[CAPACITY];
        float out[CAPACITY], expected[CAPACITY], incremental[CAPACITY], outer[CAPACITY];
        CHECK(floats(item, "x", x) == n && floats(item, "weights", weights) == kernel);
        CHECK(floats(item, "initial", initial) == history && floats(item, "final", final) == history);
        CHECK(floats(item, "expected", expected) == n);
        memcpy(state, initial, history * sizeof(float));
        state[history] = 123; out[n] = 123;
        CHECK(inkling_conv4(x, n, tokens, channels, weights, kernel, state, history, out, n));
        CHECK(state[history] == 123 && out[n] == 123);
        close_values(root, "conv", out, expected, n);
        CHECK(memcmp(state, final, history * sizeof(float)) == 0);
        memcpy(state, initial, history * sizeof(float));
        for (size_t t = 0; t < tokens; t++) {
            CHECK(inkling_conv4(x + t * channels, channels, 1, channels, weights, kernel,
                                state, history, incremental + t * channels, channels));
        }
        CHECK(memcmp(incremental, out, n * sizeof(float)) == 0);
        CHECK(memcmp(state, final, history * sizeof(float)) == 0);
        for (size_t split = 1; split < tokens; split++) {
            memcpy(state, initial, history * sizeof(float));
            size_t first = split * channels;
            CHECK(inkling_conv4(x, first, split, channels, weights, kernel, state, history, incremental, first));
            CHECK(inkling_conv4(x + first, n - first, tokens - split, channels, weights, kernel,
                                state, history, incremental + first, n - first));
            CHECK(memcmp(incremental, out, n * sizeof(float)) == 0);
            CHECK(memcmp(state, final, history * sizeof(float)) == 0);
        }
        /* Reset/reuse and exact in-place operation preserve original input history. */
        for (size_t repeat = 0; repeat < 2; repeat++) {
            memcpy(state, initial, history * sizeof(float));
            memcpy(incremental, x, n * sizeof(float));
            CHECK(inkling_conv4(incremental, n, tokens, channels, weights, kernel,
                                state, history, incremental, n));
            CHECK(memcmp(incremental, out, n * sizeof(float)) == 0);
            CHECK(memcmp(state, final, history * sizeof(float)) == 0);
        }
        CHECK(floats(item, "outer", outer) == n && floats(item, "with_outer", expected) == n);
        for (size_t i = 0; i < n; i++) out[i] += outer[i];
        close_values(root, "conv", out, expected, n);
    }
}

static void invalid_inputs(void)
{
    float x[32], out[32], weights[32], state[12];
    uint16_t bf16[32];
    uint8_t packed[16] = {0}, scales[2] = {56, 56};
    float globals[1] = {1};
    size_t indices[4] = {99, 99, 99, 99};
    for (size_t i = 0; i < 32; i++) { x[i] = 1; out[i] = 123; weights[i] = 1; bf16[i] = 0x3f80; }
    for (size_t i = 0; i < 12; i++) state[i] = 0;
    CHECK(!inkling_bf16_matvec(NULL, 32, 2, 16, x, 16, out, 2));
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, NULL, 16, out, 2));
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 16, NULL, 2));
    CHECK(!inkling_bf16_matvec(bf16, 31, 2, 16, x, 16, out, 2));
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 15, out, 2));
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 16, out, 1));
    CHECK(!inkling_bf16_matvec(bf16, 32, 0, 16, x, 16, out, 2));
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 0, x, 16, out, 2));
    CHECK(!inkling_bf16_matvec(bf16, SIZE_MAX, SIZE_MAX, 2, x, SIZE_MAX, out, SIZE_MAX));
    CHECK(!inkling_bf16_matvec(bf16, SIZE_MAX, 1, SIZE_MAX / 2, x, SIZE_MAX, out, 1));

    uint64_t original[3] = {1, 2, 16}, ps[3] = {1, 2, 8}, ss[3] = {1, 2, 1};
    InklingNvfp4Layout layout, bad;
    CHECK(inkling_nvfp4_layout(original, ps, ss, 1, &layout));
#define NV(l,e,p,pn,s,sn,g,gn,v,vn,o,on) inkling_nvfp4_matvec(l,e,p,pn,s,sn,g,gn,v,vn,o,on)
    CHECK(!NV(NULL, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 1, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 0, NULL, 16, scales, 2, globals, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, NULL, 2, globals, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, NULL, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, NULL, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, NULL, 2));
    CHECK(!NV(&layout, 0, packed, 15, scales, 2, globals, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 1, globals, 1, x, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 0, x, 16, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 15, out, 2));
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 1));
    bad = layout; bad.scale_bytes++;
    CHECK(!NV(&bad, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    bad = layout; bad.packed_bytes++;
    CHECK(!NV(&bad, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    bad = layout; bad.scale2_bytes++;
    CHECK(!NV(&bad, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    bad = layout; bad.columns = 17;
    CHECK(!NV(&bad, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    bad = layout; bad.rows = 0;
    CHECK(!NV(&bad, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    bad = layout; bad.experts = SIZE_MAX;
    CHECK(!NV(&bad, 0, packed, SIZE_MAX, scales, SIZE_MAX, globals, SIZE_MAX, x, SIZE_MAX, out, SIZE_MAX));

    CHECK(!inkling_rmsnorm(NULL, 4, weights, 4, 1e-6f, out, 4));
    CHECK(!inkling_rmsnorm(x, 4, NULL, 4, 1e-6f, out, 4));
    CHECK(!inkling_rmsnorm(x, 4, weights, 4, 1e-6f, NULL, 4));
    CHECK(!inkling_rmsnorm(x, 4, weights, 3, 1e-6f, out, 4));
    CHECK(!inkling_rmsnorm(x, 4, weights, 4, 1e-6f, out, 3));
    CHECK(!inkling_rmsnorm(x, 0, weights, 4, 1e-6f, out, 4));
    CHECK(!inkling_rmsnorm(x, SIZE_MAX, weights, SIZE_MAX, 1e-6f, out, SIZE_MAX));
    const float bad_epsilon[] = {0, -1, NAN, INFINITY};
    for (size_t i = 0; i < 4; i++) CHECK(!inkling_rmsnorm(x, 4, weights, 4, bad_epsilon[i], out, 4));
    int (*vectors[])(const float *, size_t, float *, size_t) = {inkling_sigmoid, inkling_silu, inkling_softmax};
    for (size_t i = 0; i < 3; i++) {
        CHECK(!vectors[i](NULL, 4, out, 4)); CHECK(!vectors[i](x, 4, NULL, 4));
        CHECK(!vectors[i](x, 4, out, 3)); CHECK(!vectors[i](x, 0, out, 4));
        CHECK(!vectors[i](x, SIZE_MAX, out, SIZE_MAX));
    }
    CHECK(!inkling_topk(NULL, 4, 2, indices, 2));
    CHECK(!inkling_topk(x, 4, 2, NULL, 2));
    CHECK(!inkling_topk(x, 4, 2, indices, 1));
    CHECK(!inkling_topk(x, 0, 1, indices, 4));
    CHECK(!inkling_topk(x, 4, 0, indices, 4));
    CHECK(!inkling_topk(x, 4, 5, indices, 4));
    CHECK(!inkling_topk(x, SIZE_MAX, 1, indices, 4));
    CHECK(!inkling_topk(x, SIZE_MAX / 8, SIZE_MAX / 8, indices, SIZE_MAX));
#define CONV(v,vn,t,c,w,wn,s,sn,o,on) inkling_conv4(v,vn,t,c,w,wn,s,sn,o,on)
    CHECK(!CONV(NULL, 8, 2, 4, weights, 16, state, 12, out, 8));
    CHECK(!CONV(x, 8, 2, 4, NULL, 16, state, 12, out, 8));
    CHECK(!CONV(x, 8, 2, 4, weights, 16, NULL, 12, out, 8));
    CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 12, NULL, 8));
    CHECK(!CONV(x, 7, 2, 4, weights, 16, state, 12, out, 8));
    CHECK(!CONV(x, 8, 2, 4, weights, 15, state, 12, out, 8));
    CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 11, out, 8));
    CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 12, out, 7));
    CHECK(!CONV(x, 8, 0, 4, weights, 16, state, 12, out, 8));
    CHECK(!CONV(x, 8, 2, 0, weights, 16, state, 12, out, 8));
    CHECK(!CONV(x, SIZE_MAX, SIZE_MAX, 2, weights, SIZE_MAX, state, SIZE_MAX, out, SIZE_MAX));
    CHECK(!CONV(x, SIZE_MAX, 1, SIZE_MAX / 8, weights, SIZE_MAX, state, SIZE_MAX, out, SIZE_MAX));
    for (size_t i = 0; i < 32; i++) CHECK(out[i] == 123);
    for (size_t i = 0; i < 12; i++) CHECK(state[i] == 0);
    for (size_t i = 0; i < 4; i++) CHECK(indices[i] == 99);

    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < 3; i++) {
        x[0] = nonfinite[i];
        CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 16, out, 2));
        CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
        CHECK(!inkling_rmsnorm(x, 4, weights, 4, 1e-6f, out, 4));
        CHECK(!inkling_sigmoid(x, 4, out, 4)); CHECK(!inkling_silu(x, 4, out, 4));
        CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 12, out, 8));
        if (i < 2) CHECK(!inkling_softmax(x, 4, out, 4));
        if (i == 0) CHECK(!inkling_topk(x, 4, 2, indices, 2));
        x[0] = 1; weights[0] = nonfinite[i];
        CHECK(!inkling_rmsnorm(x, 4, weights, 4, 1e-6f, out, 4));
        CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 12, out, 8));
        weights[0] = 1; state[0] = nonfinite[i];
        CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 12, out, 8));
        state[0] = 0; globals[0] = nonfinite[i];
        CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
        globals[0] = 1;
    }
    for (size_t i = 0; i < 4; i++) x[i] = -INFINITY;
    CHECK(!inkling_softmax(x, 4, out, 4));
    for (size_t i = 0; i < 32; i++) x[i] = 1;
    globals[0] = 0;
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    globals[0] = -1;
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    globals[0] = 1;
    scales[0] = 127;
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    scales[0] = 184;
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    scales[0] = 56;
    bf16[0] = 0x7f80;
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 16, out, 2));
    bf16[0] = 0x7fc0;
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 16, out, 2));
    bf16[0] = 0x4000; x[0] = FLT_MAX;
    CHECK(!inkling_bf16_matvec(bf16, 32, 2, 16, x, 16, out, 2));
    packed[0] = 0x77;
    CHECK(!NV(&layout, 0, packed, 16, scales, 2, globals, 1, x, 16, out, 2));
    CHECK(!CONV(x, 8, 2, 4, weights, 16, state, 12, out, 8));
    /* Wider RMS accumulation handles finite extremes without squaring overflow. */
    CHECK(inkling_rmsnorm(x, 1, weights, 1, 1e-6f, out, 1) && out[0] == 1);
    x[0] = 10; x[1] = 0; weights[0] = FLT_MAX;
    CHECK(!inkling_rmsnorm(x, 2, weights, 2, 1e-6f, out, 2));
#undef CONV
#undef NV
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    InklingJsonDocument document = {0};
    CHECK(inkling_json_load_file(argv[1], 512 * 1024, &document));
    bf16_cases(document.root);
    nvfp4_cases(document.root);
    vector_cases(document.root);
    conv_cases(document.root);
    invalid_inputs();
    inkling_json_document_free(&document);
    puts("Scalar math: 43 independent cases, aliases, chunk/state parity and invalid-input checks passed");
    return 0;
}
