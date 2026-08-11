#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"

static int approx(float actual, float expected)
{
    return fabsf(actual - expected) <= 1e-6f * (1.0f + fabsf(expected));
}

/* Read a whole tensor by name and decode it to F32. */
static int load_f32(
    const char *path,
    const char *header,
    uint64_t header_size,
    const char *name,
    InklingDataType expected_dtype,
    float *out,
    size_t out_count
)
{
    InklingTensorInfo tensor;

    if (!inkling_safetensors_find_tensor(header, name, &tensor)) {
        fprintf(stderr, "could not find %s\n", name);
        return 0;
    }

    if (tensor.dtype != expected_dtype) {
        fprintf(stderr, "%s has unexpected dtype %s\n",
                name, inkling_dtype_name(tensor.dtype));
        return 0;
    }

    size_t byte_count = (size_t)(tensor.data_end - tensor.data_start);
    unsigned char *raw = malloc(byte_count);

    if (raw == NULL) {
        return 0;
    }

    if (!inkling_safetensors_read_tensor_data(
            path, header_size, &tensor, raw, byte_count) ||
        !inkling_safetensors_decode_to_f32(
            &tensor, raw, byte_count, out, out_count)) {
        free(raw);
        return 0;
    }

    free(raw);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s <tiny.safetensors> <real-header.safetensors>"
            " <encoders.safetensors> <nvfp4.safetensors>\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    char *header = NULL;
    uint64_t header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[1],
            &header,
            &header_size)) {
        return EXIT_FAILURE;
    }

    InklingTensorInfo tensor;

    if (!inkling_safetensors_find_tensor(
            header,
            "test.weight",
            &tensor)) {
        fputs("could not parse test.weight\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    if (tensor.dtype != INKLING_DTYPE_F32 ||
        tensor.rank != 1 ||
        tensor.shape[0] != 1 ||
        tensor.data_start != 0 ||
        tensor.data_end != 4) {
        fputs("incorrect tensor metadata\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    uint64_t file_offset =
        8 + header_size + tensor.data_start;

    unsigned char tensor_bytes[4];

    if (!inkling_safetensors_read_tensor_data(
            argv[1],
            header_size,
            &tensor,
            tensor_bytes,
            sizeof(tensor_bytes))) {
        free(header);
        return EXIT_FAILURE;
    }

    uint32_t bits =
        (uint32_t)tensor_bytes[0] |
        (uint32_t)tensor_bytes[1] << 8 |
        (uint32_t)tensor_bytes[2] << 16 |
        (uint32_t)tensor_bytes[3] << 24;

    float value = 0.0f;

    if (sizeof(value) != sizeof(bits)) {
        fputs("this platform does not use 32-bit float\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    memcpy(&value, &bits, sizeof(value));

    if (value != 1.5f) {
        fprintf(stderr, "expected 1.5, got %.9g\n", value);
        free(header);
        return EXIT_FAILURE;
    }

    puts("Tensor metadata OK");
    puts("name:         test.weight");
    puts("dtype:        F32");

    printf("rank:         %" PRIu32 "\n", tensor.rank);
    printf("shape:        [%" PRIu64 "]\n", tensor.shape[0]);

    printf(
        "data offsets: [%" PRIu64 ", %" PRIu64 ")\n",
        tensor.data_start,
        tensor.data_end
    );

    printf("file offset:  %" PRIu64 "\n", file_offset);

    printf(
        "raw bytes:    %02x %02x %02x %02x\n",
        tensor_bytes[0],
        tensor_bytes[1],
        tensor_bytes[2],
        tensor_bytes[3]
    );

    printf("decoded F32:  %.1f\n", value);

    free(header);

    header = NULL;
    header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[2],
            &header,
            &header_size)) {
        return EXIT_FAILURE;
    }

    if (header_size != 5816) {
        fprintf(
            stderr,
            "expected 5816-byte real header, got %" PRIu64 "\n",
            header_size
        );
        free(header);
        return EXIT_FAILURE;
    }

    if (!inkling_safetensors_find_tensor(
            header,
            "model.llm.layers.0.attn.k_norm.weight",
            &tensor)) {
        fputs("could not parse real Inkling tensor\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    if (tensor.dtype != INKLING_DTYPE_BF16 ||
        tensor.rank != 1 ||
        tensor.shape[0] != 128 ||
        tensor.data_start != 3084 ||
        tensor.data_end != 3340) {
        fputs("incorrect real Inkling tensor metadata\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    uint64_t payload_size =
        tensor.data_end - tensor.data_start;

    file_offset = 8 + header_size + tensor.data_start;

    puts("Real Inkling metadata OK");
    puts("name:         model.llm.layers.0.attn.k_norm.weight");
    puts("dtype:        BF16");

    printf("shape:        [%" PRIu64 "]\n", tensor.shape[0]);
    printf("payload size: %" PRIu64 " bytes\n", payload_size);
    printf("file offset:  %" PRIu64 "\n", file_offset);

    free(header);

    /* ---- encoder deserialization (audio + vision) ---- */

    header = NULL;
    header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[3], &header, &header_size)) {
        return EXIT_FAILURE;
    }

    struct {
        const char *prefix;
        size_t expected;
    } prefixes[] = {
        {"model.audio.", 2},
        {"model.visual.", 2},
        {"model.llm.", 1},
    };

    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        char **names = NULL;
        size_t count = 0;

        if (!inkling_safetensors_enumerate_prefix(
                header, prefixes[i].prefix, &names, &count)) {
            fprintf(stderr, "enumerate failed for %s\n",
                    prefixes[i].prefix);
            free(header);
            return EXIT_FAILURE;
        }

        if (count != prefixes[i].expected) {
            fprintf(stderr, "%s: expected %zu tensors, got %zu\n",
                    prefixes[i].prefix, prefixes[i].expected, count);
            inkling_safetensors_free_names(names, count);
            free(header);
            return EXIT_FAILURE;
        }

        inkling_safetensors_free_names(names, count);
    }

    float audio_weight[6];
    float audio_expected[6] = {1.0f, -2.0f, 0.5f, 1.5f, -0.5f, 3.0f};

    if (!load_f32(argv[3], header, header_size,
                  "model.audio.encoder.weight",
                  INKLING_DTYPE_BF16, audio_weight, 6)) {
        free(header);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < 6; i++) {
        if (!approx(audio_weight[i], audio_expected[i])) {
            fprintf(stderr, "audio.encoder[%zu] = %.6g, want %.6g\n",
                    i, audio_weight[i], audio_expected[i]);
            free(header);
            return EXIT_FAILURE;
        }
    }

    float audio_norm[3];
    float audio_norm_expected[3] = {1.0f, 2.0f, 3.0f};

    if (!load_f32(argv[3], header, header_size,
                  "model.audio.final_norm.weight",
                  INKLING_DTYPE_F32, audio_norm, 3)) {
        free(header);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < 3; i++) {
        if (!approx(audio_norm[i], audio_norm_expected[i])) {
            fprintf(stderr, "audio.final_norm[%zu] mismatch\n", i);
            free(header);
            return EXIT_FAILURE;
        }
    }

    puts("");
    puts("Encoder deserialization OK");
    puts("audio:  model.audio.* -> 2 tensors, BF16 weight decoded");
    puts("vision: model.visual.* -> 2 tensors");
    puts("prefix filter excludes model.llm.*");

    free(header);

    /* ---- NVFP4 dequantization ---- */

    header = NULL;
    header_size = 0;

    if (!inkling_safetensors_read_header(
            argv[4], &header, &header_size)) {
        return EXIT_FAILURE;
    }

    InklingTensorInfo packed_info;
    InklingTensorInfo scale_info;

    if (!inkling_safetensors_find_tensor(
            header, "w.weight", &packed_info) ||
        packed_info.dtype != INKLING_DTYPE_U8 ||
        !inkling_safetensors_find_tensor(
            header, "w.weight.scale", &scale_info) ||
        scale_info.dtype != INKLING_DTYPE_F8_E4M3) {
        fputs("could not read NVFP4 tensors\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    unsigned char packed[16];
    unsigned char scales[2];

    if (!inkling_safetensors_read_tensor_data(
            argv[4], header_size, &packed_info, packed, sizeof(packed)) ||
        !inkling_safetensors_read_tensor_data(
            argv[4], header_size, &scale_info, scales, sizeof(scales))) {
        free(header);
        return EXIT_FAILURE;
    }

    float global_scale = 0.0f;

    if (!load_f32(argv[4], header, header_size,
                  "w.weight.scale2",
                  INKLING_DTYPE_F32, &global_scale, 1) ||
        !approx(global_scale, 0.5f)) {
        fputs("bad NVFP4 global scale\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    float dequant[32];

    if (!inkling_nvfp4_dequant(
            packed, sizeof(packed), scales, sizeof(scales),
            16, global_scale, dequant, 32)) {
        fputs("NVFP4 dequant failed\n", stderr);
        free(header);
        return EXIT_FAILURE;
    }

    /* spot-checks: (index, expected) covering both blocks + signs */
    struct { size_t index; float value; } checks[] = {
        {0, 0.0f}, {1, 0.25f}, {7, 3.0f}, {9, -0.25f},
        {15, -3.0f}, {17, 0.5f}, {23, 6.0f}, {31, -6.0f},
    };

    for (size_t i = 0; i < sizeof(checks) / sizeof(checks[0]); i++) {
        if (!approx(dequant[checks[i].index], checks[i].value)) {
            fprintf(stderr, "nvfp4[%zu] = %.6g, want %.6g\n",
                    checks[i].index, dequant[checks[i].index],
                    checks[i].value);
            free(header);
            return EXIT_FAILURE;
        }
    }

    puts("");
    puts("NVFP4 dequantization OK");
    puts("layout:  U8 codes + F8_E4M3 block scale + F32 global");
    puts("block:   16 elements/scale, 2 blocks");
    printf("sample:  [%.2f, %.2f, %.2f, ...  %.2f]\n",
           dequant[0], dequant[1], dequant[7], dequant[31]);

    free(header);
    return EXIT_SUCCESS;
}
