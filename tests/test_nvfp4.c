#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/inkling_nvfp4.h"
#include "../src/io/inkling_json.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

static const InklingJsonValue *field(const InklingJsonValue *value, const char *name)
{
    const InklingJsonValue *result = inkling_json_object_get(value, name);
    CHECK(result != NULL);
    return result;
}

static uint64_t integer(const InklingJsonValue *value)
{
    uint64_t result = 0;
    CHECK(inkling_json_number_u64(value, &result));
    return result;
}

static uint32_t float_bits(float value)
{
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static uint32_t little_u32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static float little_float(const uint8_t *bytes)
{
    uint32_t bits = little_u32(bytes);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void shape(const InklingJsonValue *value, uint64_t result[3])
{
    CHECK(inkling_json_array_size(value) == 3);
    for (size_t i = 0; i < 3; i++) result[i] = integer(inkling_json_array_at(value, i));
}

static void unhex(const InklingJsonValue *record, uint8_t *bytes, size_t count)
{
    size_t length = 0;
    const char *hex = inkling_json_string(field(record, "hex"), &length);
    CHECK(hex != NULL && length == count * 2);
    CHECK(integer(field(record, "length")) == count);
    const char *digits = "0123456789abcdef";
    for (size_t i = 0; i < count; i++) {
        const char *hi = strchr(digits, hex[i * 2]);
        const char *lo = strchr(digits, hex[i * 2 + 1]);
        CHECK(hi != NULL && lo != NULL && *hi != '\0' && *lo != '\0');
        bytes[i] = (uint8_t)((hi - digits) * 16 + (lo - digits));
    }
}

static void tables(const InklingJsonValue *root)
{
    const InklingJsonValue *table = field(field(root, "tables"), "e2m1_f32_bits");
    CHECK(inkling_json_array_size(table) == 16);
    for (size_t code = 0; code < 16; code++) {
        CHECK(float_bits(inkling_e2m1_to_f32((uint8_t)code)) ==
              integer(inkling_json_array_at(table, code)));
    }
    CHECK(isnan(inkling_e2m1_to_f32(16)) && isnan(inkling_e2m1_to_f32(255)));
    table = field(field(root, "tables"), "e4m3_f32_bits");
    CHECK(inkling_json_array_size(table) == 256);
    for (size_t code = 0; code < 256; code++) {
        CHECK(float_bits(inkling_e4m3_to_f32((uint8_t)code)) ==
              integer(inkling_json_array_at(table, code)));
    }
    CHECK(inkling_e4m3_to_f32(0x7e) == 448.0f);
    CHECK(inkling_e4m3_to_f32(1) == 0x1p-9f);
}

static void real_samples(const InklingJsonValue *root)
{
    const InklingJsonValue *projections = field(root, "projections");
    CHECK(inkling_json_array_size(projections) == 2);
    size_t swapped_differences = 0, scale_differences = 0;
    size_t inverse_differences = 0, expert_differences = 0;
    for (size_t projection = 0; projection < 2; projection++) {
        const InklingJsonValue *p = inkling_json_array_at(projections, projection);
        const char *name = inkling_json_string(field(p, "weight"), NULL);
        CHECK(name != NULL && strlen(name) < 200);
        char scale_name[256];
        snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
        uint64_t original[3], packed_shape[3], scales_shape[3];
        shape(field(p, "logical_shape"), original);
        shape(field(field(field(root, "tensors"), name), "shape"), packed_shape);
        shape(field(field(field(root, "tensors"), scale_name), "shape"), scales_shape);
        InklingNvfp4Layout layout;
        CHECK(inkling_nvfp4_layout(original, packed_shape, scales_shape, 256, &layout));
        uint8_t globals[1024];
        unhex(field(p, "scale2"), globals, sizeof(globals));
        CHECK(layout.scale2_bytes == sizeof(globals));
        const InklingJsonValue *samples = field(p, "samples");
        CHECK(inkling_json_array_size(samples) == 9);
        for (size_t sample = 0; sample < 9; sample++) {
            const InklingJsonValue *s = inkling_json_array_at(samples, sample);
            size_t expert = (size_t)integer(field(s, "expert"));
            size_t row = (size_t)integer(field(s, "row"));
            size_t block = (size_t)integer(field(s, "block"));
            uint8_t packed[16], scales[2];
            unhex(field(s, "packed"), packed, sizeof(packed));
            unhex(field(s, "scales"), scales, sizeof(scales));
            const InklingJsonValue *expected = field(s, "expected_f32_bits");
            CHECK(inkling_json_array_size(expected) == 32);
            for (size_t b = 0; b < 2; b++) {
                InklingNvfp4Offsets offsets;
                CHECK(inkling_nvfp4_offsets(&layout, expert, row, block + b, &offsets));
                CHECK(offsets.packed == integer(field(field(s, "packed"), "tensor_offset")) + b * 8);
                CHECK(offsets.scale == integer(field(field(s, "scales"), "tensor_offset")) + b);
                CHECK(offsets.scale2 == expert * 4);
                float global = little_float(globals + offsets.scale2);
                float actual[16], swapped[16], wrong_scale[16], inverse[16], wrong_expert[16];
                CHECK(inkling_nvfp4_decode_block(packed + b * 8, 8, scales[b], global, actual, 16));
                uint8_t exchanged[8];
                for (size_t i = 0; i < 8; i++) {
                    uint8_t byte = packed[b * 8 + i];
                    exchanged[i] = (uint8_t)((byte >> 4) | ((unsigned int)byte << 4));
                }
                CHECK(inkling_nvfp4_decode_block(exchanged, 8, scales[b], global, swapped, 16));
                CHECK(inkling_nvfp4_decode_block(packed + b * 8, 8, scales[1 - b], global, wrong_scale, 16));
                CHECK(inkling_nvfp4_decode_block(packed + b * 8, 8, scales[b], 1.0f / global, inverse, 16));
                CHECK(inkling_nvfp4_decode_block(packed + b * 8, 8, scales[b], little_float(globals), wrong_expert, 16));
                for (size_t i = 0; i < 16; i++) {
                    CHECK(float_bits(actual[i]) == integer(inkling_json_array_at(expected, b * 16 + i)));
                    swapped_differences += float_bits(actual[i]) != float_bits(swapped[i]) ? 1U : 0U;
                    scale_differences += float_bits(actual[i]) != float_bits(wrong_scale[i]) ? 1U : 0U;
                    inverse_differences += float_bits(actual[i]) != float_bits(inverse[i]) ? 1U : 0U;
                    expert_differences += float_bits(actual[i]) != float_bits(wrong_expert[i]) ? 1U : 0U;
                }
            }
        }
    }
    CHECK(swapped_differences > 0 && scale_differences > 0 && inverse_differences > 0 && expert_differences > 0);
}

static void invalid_inputs(void)
{
    uint64_t original[3] = {3, 2, 32}, packed[3] = {3, 2, 16}, scales[3] = {3, 2, 2};
    InklingNvfp4Layout layout;
    CHECK(inkling_nvfp4_layout(original, packed, scales, 3, &layout));
    CHECK(layout.packed_bytes == 96 && layout.scale_bytes == 12 && layout.scale2_bytes == 12);
    InklingNvfp4Offsets offsets;
    CHECK(inkling_nvfp4_offsets(&layout, 2, 1, 1, &offsets));
    CHECK(offsets.packed == 88 && offsets.scale == 11 && offsets.scale2 == 8);
    CHECK(!inkling_nvfp4_offsets(&layout, 3, 0, 0, &offsets));
    CHECK(offsets.packed == 0 && offsets.scale == 0 && offsets.scale2 == 0);
    CHECK(!inkling_nvfp4_offsets(&layout, 0, 2, 0, &offsets));
    CHECK(!inkling_nvfp4_offsets(&layout, 0, 0, 2, &offsets));
    CHECK(!inkling_nvfp4_offsets(NULL, 0, 0, 0, &offsets));
    CHECK(!inkling_nvfp4_offsets(&layout, 0, 0, 0, NULL));
    CHECK(!inkling_nvfp4_layout(original, packed, scales, 2, &layout));
    CHECK(layout.experts == 0 && layout.packed_bytes == 0);
    scales[2] = 3; /* Padded scale rows must not be guessed. */
    CHECK(!inkling_nvfp4_layout(original, packed, scales, 3, &layout));
    scales[2] = 2;
    packed[2] = 17;
    CHECK(!inkling_nvfp4_layout(original, packed, scales, 3, &layout));
    packed[2] = 16;
    original[2] = 33; /* Same integer quotients, incomplete last block. */
    CHECK(!inkling_nvfp4_layout(original, packed, scales, 3, &layout));
    original[2] = 0;
    CHECK(!inkling_nvfp4_layout(original, packed, scales, 3, &layout));
    original[2] = 32;
    original[0] = packed[0] = scales[0] = UINT64_MAX;
    CHECK(!inkling_nvfp4_layout(original, packed, scales, UINT64_MAX, &layout));
    CHECK(!inkling_nvfp4_layout(NULL, packed, scales, 3, &layout));
    CHECK(!inkling_nvfp4_layout(original, NULL, scales, 3, &layout));
    CHECK(!inkling_nvfp4_layout(original, packed, NULL, 3, &layout));
    CHECK(!inkling_nvfp4_layout(original, packed, scales, 3, NULL));
    layout = (InklingNvfp4Layout){SIZE_MAX, SIZE_MAX, 16, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    CHECK(!inkling_nvfp4_offsets(&layout, SIZE_MAX - 1, 0, 0, &offsets));

    uint8_t bytes[8] = {0x77};
    float out[16];
    for (size_t i = 0; i < 16; i++) out[i] = 123.0f;
    CHECK(!inkling_nvfp4_decode_block(NULL, 8, 0x38, 1, out, 16));
    CHECK(!inkling_nvfp4_decode_block(bytes, 8, 0x38, 1, NULL, 16));
    CHECK(!inkling_nvfp4_decode_block(bytes, 7, 0x38, 1, out, 16));
    CHECK(!inkling_nvfp4_decode_block(bytes, 8, 0x38, 1, out, 15));
    CHECK(!inkling_nvfp4_decode_block(bytes, 8, 0x7f, 1, out, 16));
    CHECK(!inkling_nvfp4_decode_block(bytes, 8, 0xb8, 1, out, 16));
    const float bad_globals[] = {0, -1, INFINITY, NAN, FLT_MAX};
    for (size_t i = 0; i < sizeof(bad_globals) / sizeof(bad_globals[0]); i++) {
        CHECK(!inkling_nvfp4_decode_block(bytes, 8, 0x38, bad_globals[i], out, 16));
    }
    CHECK(!inkling_nvfp4_decode_block(bytes, 8, 0x7e, FLT_MAX, out, 16));
    for (size_t i = 0; i < 16; i++) CHECK(out[i] == 123.0f);
    CHECK(inkling_nvfp4_decode_block(bytes, 8, 0, 1, out, 16));
    for (size_t i = 0; i < 16; i++) CHECK(out[i] == 0);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    InklingJsonDocument document = {0};
    CHECK(inkling_json_load_file(argv[1], 256 * 1024, &document));
    tables(document.root);
    real_samples(document.root);
    invalid_inputs();
    inkling_json_document_free(&document);
    puts("NVFP4: all E2M1/E4M3 codes, 18 real samples (576 values), layout and failure tests passed");
    return 0;
}
