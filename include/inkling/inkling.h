#ifndef INKLING_INKLING_H
#define INKLING_INKLING_H
#define INKLING_MAX_TENSOR_RANK 8
#define INKLING_INDEX_MAX_NAME_LENGTH 128
#define INKLING_INDEX_MAX_SHARD_LENGTH 64

#include <stddef.h>
#include <stdint.h>

typedef enum {
    INKLING_DTYPE_UNKNOWN = 0,
    INKLING_DTYPE_F32,
    INKLING_DTYPE_BF16,
    INKLING_DTYPE_I64,
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
    uint32_t *local_layer_ids;
    size_t num_local_layers;
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

/* On success, owns local_layer_ids; release with inkling_config_free before
 * reloading or discarding. Failure leaves config unchanged. Do not free copies. */
int inkling_config_load(const char *path, InklingConfig *config);
int inkling_config_is_valid(const InklingConfig *config);
void inkling_config_free(InklingConfig *config);

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

typedef struct {
    char name[INKLING_INDEX_MAX_NAME_LENGTH];
    char shard[INKLING_INDEX_MAX_SHARD_LENGTH];
    InklingDataType dtype;
    uint32_t rank;
    uint64_t shape[INKLING_MAX_TENSOR_RANK];
    uint64_t data_offset; /* Relative to 8 + header_size, not the file start. */
    uint64_t byte_length;
    uint32_t shard_id;
} InklingTensor;

typedef InklingTensor InklingIndexEntry;

typedef struct {
    InklingIndexEntry *entries;
    uint64_t count;
    uint64_t capacity;
    uint64_t total_size;
    uint64_t *slots; /* Entry index + 1; zero is an empty slot. */
    size_t slot_count;
    uint32_t num_shards;
} InklingIndex;

/* The index is the tensor catalogue. Metadata starts UNKNOWN until a shard is
 * bound. Returned pointers remain valid until inkling_index_free(). */
const InklingTensor *inkling_index_find_tensor(
    const InklingIndex *index, const char *tensor_name
);

/* Explicit-size entry point for metadata fixtures / callers with a trusted
 * shard length. No payload is read. Failure leaves the catalogue unchanged. */
int inkling_index_bind_header(
    InklingIndex *index, const char *shard, const char *header_json,
    uint64_t header_size, uint64_t shard_size
);

/* Production entry point: checks the actual shard length. */
int inkling_index_load_shard(
    InklingIndex *index, const char *shard, const char *path
);

/* Parse exactly header_size bytes and validate all tensor ranges against
 * shard_size (including the 8-byte prefix and header). Caller owns *tensors.
 * Outputs are empty on failure. U8 is one stored byte (two packed FP4 values
 * when used by this checkpoint); original_shape carries the logical shape. */
int inkling_safetensors_parse_header(
    const char *header_json, uint64_t header_size, uint64_t shard_size,
    InklingTensor **tensors, size_t *count
);

int inkling_safetensors_load_shard(
    const char *path, InklingTensor **tensors, size_t *count
);

/* On success owns entries and slots; free before reloading. Failure leaves the
 * caller's index unchanged. Shard IDs follow first appearance in the index. */
int inkling_index_load(
    const char *path,
    InklingIndex *index
);

int inkling_index_find_shard(
    const InklingIndex *index,
    const char *tensor_name,
    char *shard_out,
    size_t shard_out_size
);

void inkling_index_free(InklingIndex *index);

#endif
