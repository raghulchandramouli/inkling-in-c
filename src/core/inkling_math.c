#include <math.h>

#include "inkling_alloc.h"
#include "inkling_bf16.h"
#include "inkling_math.h"

static int fits(size_t count, size_t width)
{
    return count > 0 && count <= (size_t)PTRDIFF_MAX / width;
}

static int finite_vector(const float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) if (!isfinite(x[i])) return 0;
    return 1;
}

static int vector_args(const float *x, size_t n, const float *out, size_t out_count)
{
    return x != NULL && out != NULL && fits(n, sizeof(float)) && out_count >= n;
}

int inkling_bf16_matvec(const uint16_t *weights, size_t weight_count,
                        size_t rows, size_t columns, const float *x,
                        size_t x_count, float *out, size_t out_count)
{
    size_t count;
    if (weights == NULL || x == NULL || out == NULL ||
        !inkling_size_mul(rows, columns, &count) || !fits(count, sizeof(*weights)) ||
        !fits(rows, sizeof(*out)) || !fits(columns, sizeof(*x)) ||
        weight_count < count || x_count < columns || out_count < rows ||
        !finite_vector(x, columns)) return 0;
    for (size_t row = 0; row < rows; row++) {
        float sum = 0;
        for (size_t col = 0; col < columns; col++)
            sum += inkling_bf16_to_f32(weights[row * columns + col]) * x[col];
        if (!isfinite(sum)) return 0;
        out[row] = sum;
    }
    return 1;
}

int inkling_nvfp4_matvec(const InklingNvfp4Layout *layout, size_t expert,
                         const uint8_t *packed, size_t packed_bytes,
                         const uint8_t *scales, size_t scale_bytes,
                         const float *scale2, size_t scale2_count,
                         const float *x, size_t x_count,
                         float *out, size_t out_count)
{
    if (layout == NULL || packed == NULL || scales == NULL || scale2 == NULL ||
        x == NULL || out == NULL) return 0;
    uint64_t shape[3] = {layout->experts, layout->rows, layout->columns};
    uint64_t packed_shape[3] = {shape[0], shape[1], shape[2] / 2};
    uint64_t scale_shape[3] = {shape[0], shape[1], shape[2] / 16};
    InklingNvfp4Layout checked;
    if (!inkling_nvfp4_layout(shape, packed_shape, scale_shape, shape[0], &checked) ||
        checked.packed_bytes != layout->packed_bytes ||
        checked.scale_bytes != layout->scale_bytes ||
        checked.scale2_bytes != layout->scale2_bytes ||
        !fits(checked.packed_bytes, 1) ||
        !fits(checked.columns, sizeof(*x)) || packed_bytes < checked.packed_bytes ||
        scale_bytes < checked.scale_bytes || scale2_count < checked.experts ||
        x_count < checked.columns || out_count < checked.rows ||
        !finite_vector(x, checked.columns)) return 0;
    InklingNvfp4Offsets offsets;
    if (!inkling_nvfp4_offsets(&checked, expert, 0, 0, &offsets)) return 0;
    for (size_t row = 0; row < checked.rows; row++) {
        float sum = 0;
        for (size_t block = 0; block < checked.columns / 16; block++) {
            float decoded[16];
            if (!inkling_nvfp4_decode_block(packed + offsets.packed, 8,
                    scales[offsets.scale], scale2[expert], decoded, 16)) return 0;
            for (size_t i = 0; i < 16; i++) sum += decoded[i] * x[block * 16 + i];
            offsets.packed += 8;
            offsets.scale++;
        }
        if (!isfinite(sum)) return 0;
        out[row] = sum;
    }
    return 1;
}

int inkling_rmsnorm(const float *x, size_t n, const float *weights,
                     size_t weight_count, float epsilon, float *out, size_t out_count)
{
    if (!vector_args(x, n, out, out_count) || weights == NULL || weight_count < n ||
        !isfinite(epsilon) || epsilon <= 0 || !finite_vector(x, n) ||
        !finite_vector(weights, n)) return 0;
    double sum = 0;
    for (size_t i = 0; i < n; i++) sum += (double)x[i] * x[i];
    double inverse = 1.0 / sqrt(sum / (double)n + epsilon);
    for (size_t i = 0; i < n; i++) {
        float normalized = (float)((double)x[i] * inverse);
        float value = weights[i] * normalized;
        if (!isfinite(value)) return 0;
        out[i] = value;
    }
    return 1;
}

static float sigmoid(float x)
{
    float exponential = expf(-fabsf(x));
    return (x >= 0 ? 1.0f : exponential) / (1.0f + exponential);
}

int inkling_sigmoid(const float *x, size_t n, float *out, size_t out_count)
{
    if (!vector_args(x, n, out, out_count) || !finite_vector(x, n)) return 0;
    for (size_t i = 0; i < n; i++) out[i] = sigmoid(x[i]);
    return 1;
}

int inkling_silu(const float *x, size_t n, float *out, size_t out_count)
{
    if (!vector_args(x, n, out, out_count) || !finite_vector(x, n)) return 0;
    for (size_t i = 0; i < n; i++) out[i] = x[i] * sigmoid(x[i]);
    return 1;
}

int inkling_softmax(const float *x, size_t n, float *out, size_t out_count)
{
    if (!vector_args(x, n, out, out_count)) return 0;
    float maximum = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        if (isnan(x[i]) || x[i] == INFINITY) return 0;
        if (x[i] > maximum) maximum = x[i];
    }
    if (maximum == -INFINITY) return 0;
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        /* Subtraction in double also handles opposite finite float extremes. */
        out[i] = (float)exp((double)x[i] - maximum);
        sum += out[i];
    }
    for (size_t i = 0; i < n; i++) out[i] = (float)(out[i] / sum);
    return 1;
}

int inkling_topk(const float *x, size_t n, size_t k,
                  size_t *indices, size_t indices_count)
{
    if (x == NULL || indices == NULL || !fits(n, sizeof(*x)) ||
        !fits(k, sizeof(*indices)) || k > n || indices_count < k) return 0;
    for (size_t i = 0; i < n; i++) if (isnan(x[i])) return 0;
    /* ponytail: O(n*k) insertion is sufficient for six routed experts;
     * use a heap if large-k selection becomes a measured bottleneck. */
    size_t used = 0;
    for (size_t i = 0; i < n; i++) {
        size_t position = 0;
        while (position < used && x[indices[position]] >= x[i]) position++;
        if (position == k) continue;
        if (used < k) used++;
        for (size_t j = used - 1; j > position; j--) indices[j] = indices[j - 1];
        indices[position] = i;
    }
    return 1;
}

int inkling_conv4(const float *x, size_t x_count, size_t tokens, size_t channels,
                   const float *weights, size_t weight_count,
                   float *state, size_t state_count, float *out, size_t out_count)
{
    size_t count, kernel_count, history_count;
    if (x == NULL || weights == NULL || state == NULL || out == NULL ||
        !inkling_size_mul(tokens, channels, &count) || !fits(count, sizeof(*x)) ||
        !inkling_size_mul(channels, 4, &kernel_count) || !fits(kernel_count, sizeof(*weights)) ||
        !inkling_size_mul(channels, 3, &history_count) ||
        x_count < count || out_count < count || weight_count < kernel_count ||
        state_count < history_count || !finite_vector(x, count) ||
        !finite_vector(weights, kernel_count) || !finite_vector(state, history_count)) return 0;
    for (size_t t = 0; t < tokens; t++) {
        for (size_t c = 0; c < channels; c++) {
            float current = x[t * channels + c];
            float *history = state + c * 3;
            const float *kernel = weights + c * 4;
            float sum = 0;
            for (size_t tap = 0; tap < 3; tap++) sum += history[tap] * kernel[tap];
            sum += current * kernel[3];
            sum += current;
            if (!isfinite(sum)) return 0;
            history[0] = history[1];
            history[1] = history[2];
            history[2] = current;
            out[t * channels + c] = sum;
        }
    }
    return 1;
}
