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
    INKLING_DTYPE_BF16
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

typedef struct {
    char name[INKLING_INDEX_MAX_NAME_LENGTH];
    char shard[INKLING_INDEX_MAX_SHARD_LENGTH];
} InklingIndexEntry;

typedef struct {
    InklingIndexEntry *entries;
    uint64_t count;
    uint64_t capacity;
    uint64_t total_size;
} InklingIndex;

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
