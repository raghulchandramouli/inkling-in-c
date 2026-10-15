#ifndef INKLING_MODEL_INTERNAL_H
#define INKLING_MODEL_INTERNAL_H
#include "inkling/inkling_model.h"
#include "inkling_nvfp4.h"
typedef struct { uint32_t shard; uint64_t start, bytes; } InklingRange;
typedef enum { INKLING_MATRIX_BF16, INKLING_MATRIX_NVFP4 } InklingMatrixKind;
typedef struct {
    size_t experts, rows, columns;
    InklingMatrixKind kind;
    InklingRange data, scales;
    float *scale2;
    InklingNvfp4Layout layout;
} InklingMatrix;
typedef struct {
    size_t original_id, heads, kv_heads, head_dim, extent, window;
    InklingMatrix q, k, v, r, o;
    float *q_norm, *k_norm, *relative, *k_conv, *v_conv;
    float *attn_norm, *mlp_norm, *attn_conv, *mlp_conv;
    int dense;
    InklingMatrix w13, down, shared_w13, shared_down, router;
    float *correction;
    float global_scale;
} InklingLayer;
typedef struct { char *path; uint64_t header_size, file_size; } InklingShard;
typedef struct { InklingRange range; char *path; } InklingBlob;
struct InklingModel {
    InklingConfig config;
    InklingIndex index;
    InklingShard *shards;
    size_t shard_count;
    InklingBlob *blobs;
    size_t blob_count;
    int extracted;
    InklingLayer *layers;
    size_t layer_count;
    int complete;
    InklingMatrix embed, unembed;
    float *embed_norm, *final_norm;
};
int inkling_model_open_text_fixture(const char *directory, InklingModel **out);
int inkling_model_open_layer_fixture(const char *metadata_directory,
                                     const char *manifest, InklingModel **out);
int inkling_model_read(const InklingModel *model, InklingRange range,
                       uint64_t relative, size_t bytes, void *destination);
typedef void (*InklingTrace)(void *user, const char *name, const float *values,
                             const size_t *indices, size_t count);
void inkling_context_trace(InklingContext *context, InklingTrace trace, void *user);
int inkling_attention_forward(InklingContext *context, const InklingLayer *layer,
                              const float *input, size_t tokens,
                              size_t position_start, float *output);
int inkling_mlp_forward(InklingContext *context, const InklingLayer *layer,
                        const float *input, size_t tokens, float *output);
int inkling_layer_forward(InklingContext *context, const InklingLayer *layer,
                          const float *input, size_t tokens, float *output);
int inkling_router_forward(const InklingModel *model, const InklingLayer *layer,
                            const float *logits, size_t *selected, float *weights);
int inkling_verify_model_quiet(const char *directory, int metadata_only);
#endif
