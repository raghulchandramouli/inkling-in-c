#ifndef INKLING_INKLING_H
#define INKLING_INKLING_H
#define INKLING_MAX_TENSOR_RANK 8

#include <stddef.h>
#include <stdint.h>

typedef enum {
    INKLING_DTYPE_UNKNOWN = 0,
    INKLING_DTYPE_F32,
    INKLING_DTYPE_BF16,
    INKLING_DTYPE_F16,
    INKLING_DTYPE_F8_E4M3,
    INKLING_DTYPE_U8
} InklingDataType;

typedef struct {
    InklingDataType dtype;

    uint32_t rank;
    uint64_t shape[INKLING_MAX_TENSOR_RANK];

    uint64_t data_start;
    uint64_t data_end;
} InklingTensorInfo;

typedef struct {
    uint32_t model_max_length;
    uint32_t vocab_size;
    uint32_t eos_token_id;

    uint32_t hidden_size;
    uint32_t num_hidden_layers;

    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;

    uint32_t sliding_window_size;
    uint32_t global_attention_stride;
    uint32_t relative_dimension;
    uint32_t relative_extent;
    uint32_t sconv_kernel_size;

    uint32_t num_routed_experts;
    uint32_t num_experts_per_token;
    uint32_t num_shared_experts;

    uint32_t dense_intermediate_size;
    uint32_t expert_intermediate_size;

    float rms_norm_epsilon;
    float route_scale;
} InklingConfig;

int inkling_config_load(const char *path, InklingConfig *config);
int inkling_config_is_valid(const InklingConfig *config);

int inkling_safetensors_header_size(
    const char *path,
    uint64_t *header_size
);

/*
 * On success, the caller owns *header_json and must free it.
 */
int inkling_safetensors_read_header(
    const char *path,
    char **header_json,
    uint64_t *header_size
);

int inkling_safetensors_find_tensor(
    const char *header_json,
    const char *tensor_name,
    InklingTensorInfo *tensor
);

int inkling_safetensors_read_tensor_data(
    const char *path,
    uint64_t header_size,
    const InklingTensorInfo *tensor,
    void *destination,
    size_t destination_size
);

/*
 * Human-readable dtype name (e.g. "BF16"), or "UNKNOWN".
 */
const char *inkling_dtype_name(InklingDataType dtype);

/*
 * Enumerate every tensor whose name begins with `prefix`
 * (e.g. "model.audio." or "model.visual.").
 *
 * On success, *names is a malloc'd array of *count malloc'd
 * strings, sorted by first appearance in the header. The caller
 * owns both and must release them with
 * inkling_safetensors_free_names(). A prefix that matches nothing
 * succeeds with *count == 0 and *names == NULL.
 */
int inkling_safetensors_enumerate_prefix(
    const char *header_json,
    const char *prefix,
    char ***names,
    size_t *count
);

void inkling_safetensors_free_names(
    char **names,
    size_t count
);

/*
 * Scalar dtype decoders. Each returns the value as a 32-bit float.
 *   - bf16 / f16 take the raw 16-bit little-endian bit pattern.
 *   - f8_e4m3 takes the raw 8-bit pattern (NaN decodes to NAN).
 */
float inkling_bf16_to_f32(uint16_t bits);
float inkling_f16_to_f32(uint16_t bits);
float inkling_f8_e4m3_to_f32(uint8_t bits);

/*
 * Decode a whole tensor payload into F32. Supports F32, BF16 and
 * F16 (the dtypes used by Inkling's audio/vision encoders and
 * norms). `raw` is the bytes returned by read_tensor_data;
 * `out_count` must equal the tensor's element count.
 */
int inkling_safetensors_decode_to_f32(
    const InklingTensorInfo *tensor,
    const void *raw,
    size_t raw_size,
    float *out,
    size_t out_count
);

/*
 * Dequantize an NVFP4 tensor to F32.
 *
 * NVFP4 (as shipped by Inkling's quantized checkpoints) stores each
 * weight as three tensors:
 *   - the packed codes: dtype U8, two E2M1 4-bit codes per byte
 *     (low nibble first), so out_count == packed_bytes * 2;
 *   - a per-block scale: dtype F8_E4M3, one value per `block_size`
 *     logical elements (block_size is 16 for Inkling);
 *   - a per-tensor global scale: a single F32 (the ".scale2" /
 *     "global_scale" entry).
 *
 * Reconstruction per element is
 *     value = e2m1(code) * e4m3(block_scale) * global_scale
 *
 * `block_scale_count` must be at least out_count / block_size.
 */
int inkling_nvfp4_dequant(
    const uint8_t *packed,
    size_t packed_bytes,
    const uint8_t *block_scale_e4m3,
    size_t block_scale_count,
    uint32_t block_size,
    float global_scale,
    float *out,
    size_t out_count
);

#endif
