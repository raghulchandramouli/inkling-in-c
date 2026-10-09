#include <math.h>
#include <string.h>

#include "inkling_alloc.h"
#include "inkling_bf16.h"
#include "inkling_nvfp4.h"

int inkling_nvfp4_layout(const uint64_t original[3], const uint64_t packed[3],
                        const uint64_t scales[3], uint64_t scale2_count,
                        InklingNvfp4Layout *out)
{
    if (out == NULL) return 0;
    *out = (InklingNvfp4Layout){0};
    if (original == NULL || packed == NULL || scales == NULL) return 0;
    for (size_t axis = 0; axis < 3; axis++) {
        if (original[axis] == 0 || original[axis] > SIZE_MAX ||
            packed[axis] != original[axis] / (axis == 2 ? 2U : 1U) ||
            scales[axis] != original[axis] / (axis == 2 ? 16U : 1U)) return 0;
    }
    if (original[2] % 16 != 0 || scale2_count != original[0]) return 0;
    InklingNvfp4Layout parsed = {0};
    parsed.experts = (size_t)original[0];
    parsed.rows = (size_t)original[1];
    parsed.columns = (size_t)original[2];
    size_t total_rows = 0;
    if (!inkling_size_mul(parsed.experts, parsed.rows, &total_rows) ||
        !inkling_size_mul(total_rows, parsed.columns / 2, &parsed.packed_bytes) ||
        !inkling_size_mul(parsed.experts, 4, &parsed.scale2_bytes)) return 0;
    parsed.scale_bytes = parsed.packed_bytes / 8;
    *out = parsed;
    return 1;
}

int inkling_nvfp4_offsets(const InklingNvfp4Layout *layout, size_t expert,
                         size_t row, size_t block, InklingNvfp4Offsets *out)
{
    if (out == NULL) return 0;
    *out = (InklingNvfp4Offsets){0};
    if (layout == NULL || layout->columns == 0 || layout->columns % 16 != 0 ||
        expert >= layout->experts || row >= layout->rows ||
        block >= layout->columns / 16) return 0;
    size_t linear = 0;
    InklingNvfp4Offsets parsed = {0};
    if (!inkling_size_mul(expert, layout->rows, &linear) ||
        !inkling_size_add(linear, row, &linear) ||
        !inkling_size_mul(linear, layout->columns / 16, &linear) ||
        !inkling_size_add(linear, block, &parsed.scale) ||
        !inkling_size_mul(parsed.scale, 8, &parsed.packed) ||
        !inkling_size_mul(expert, 4, &parsed.scale2) ||
        layout->packed_bytes < 8 || parsed.packed > layout->packed_bytes - 8 ||
        parsed.scale >= layout->scale_bytes || layout->scale2_bytes < 4 ||
        parsed.scale2 > layout->scale2_bytes - 4) return 0;
    *out = parsed;
    return 1;
}

float inkling_e2m1_to_f32(uint8_t code)
{
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return code < 16 ? values[code] : NAN;
}

float inkling_e4m3_to_f32(uint8_t code)
{
    unsigned int exponent = (code >> 3) & 15U;
    unsigned int mantissa = code & 7U;
    uint16_t sign = (uint16_t)((unsigned int)(code & 128U) << 8);
    if (exponent == 0) {
        float value = (float)mantissa * 0x1p-9f;
        return sign != 0 ? -value : value;
    }
    uint16_t bits = (uint16_t)(sign | ((exponent + 120U) << 7) | (mantissa << 4));
    if ((code & 127U) == 127U) bits = (uint16_t)(sign | 0x7ff0U);
    return inkling_bf16_to_f32(bits);
}

int inkling_nvfp4_decode_block(const uint8_t *packed, size_t packed_bytes,
                              uint8_t scale, float scale2, float *out,
                              size_t out_count)
{
    float local_scale = inkling_e4m3_to_f32(scale);
    if (packed == NULL || out == NULL || packed_bytes < 8 || out_count < 16 ||
        !isfinite(local_scale) || local_scale < 0 ||
        !isfinite(scale2) || scale2 <= 0) return 0;
    float combined_scale = local_scale * scale2;
    float decoded[16];
    for (size_t i = 0; i < 16; i++) {
        uint8_t code = (uint8_t)((packed[i / 2] >> ((i % 2) * 4)) & 15);
        decoded[i] = inkling_e2m1_to_f32(code) * combined_scale;
        if (!isfinite(decoded[i])) return 0;
    }
    memcpy(out, decoded, sizeof(decoded));
    return 1;
}
