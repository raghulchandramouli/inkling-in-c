#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "inkling_model_internal.h"
#include "inkling_alloc.h"
#include "inkling_bf16.h"
#include "inkling_json.h"

#define REVISION "b6a99534467840620d411e4cd4ad5819b2610d9c"

static int fail(const char *message)
{
    fprintf(stderr, "model: %s\n", message);
    return 0;
}

static char *join(const char *directory, const char *name)
{
    size_t length, total;
    if (!directory || !name || !inkling_size_add(strlen(directory), strlen(name), &length) ||
        !inkling_size_add(length, 2, &total)) return NULL;
    char *path = malloc(total);
    if (path) snprintf(path, total, "%s/%s", directory, name);
    return path;
}

static int file_size(const char *path, uint64_t *size)
{
    struct stat status;
    if (stat(path, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0) {
        fprintf(stderr, "model: cannot stat regular file %s\n", path);
        return 0;
    }
    *size = (uint64_t)status.st_size;
    return 1;
}

static int read_file(const char *path, uint64_t offset, size_t bytes, void *destination)
{
    if (offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset) return fail("file offset overflow");
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "model: cannot open %s: %s\n", path, strerror(errno));
        return 0;
    }
#ifdef _WIN32
    int ok = _fseeki64(file, (int64_t)offset, SEEK_SET) == 0;
#else
    off_t position = (off_t)offset;
    int ok = position >= 0 && (uint64_t)position == offset && fseeko(file, position, SEEK_SET) == 0;
#endif
    if (ok) ok = fread(destination, 1, bytes, file) == bytes;
    if (fclose(file) != 0) ok = 0;
    if (!ok) fprintf(stderr, "model: short or failed read from %s at byte %" PRIu64 " (%zu bytes)\n", path, offset, bytes);
    return ok;
}

int inkling_model_read(const InklingModel *model, InklingRange range,
                       uint64_t relative, size_t bytes, void *destination)
{
    if (!model || (bytes && !destination) || range.shard >= model->shard_count ||
        relative > range.bytes || bytes > range.bytes - relative ||
        range.start > UINT64_MAX - range.bytes) return fail("invalid tensor read bounds");
    const InklingShard *shard = &model->shards[range.shard];
    if (shard->header_size > UINT64_MAX - 8 || shard->file_size < shard->header_size + 8 ||
        range.start + range.bytes > shard->file_size - shard->header_size - 8)
        return fail("tensor read exceeds original shard");
    if (!bytes) return 1;
    int handled;
    if (!inkling_packed_read(model, range, relative, bytes, destination, &handled)) return 0;
    if (handled) return 1;
    uint64_t start = range.start + relative;
    if (!model->extracted) {
        if (!shard->path || start > UINT64_MAX - shard->header_size - 8) return fail("unavailable shard");
        return read_file(shard->path, 8 + shard->header_size + start, bytes, destination);
    }
    unsigned char *output = destination;
    size_t remaining = bytes;
    for (size_t i = 0; remaining && i < model->blob_count; i++) {
        const InklingBlob *blob = &model->blobs[i];
        if (blob->range.shard != range.shard || start < blob->range.start ||
            start - blob->range.start >= blob->range.bytes) continue;
        uint64_t offset = start - blob->range.start;
        uint64_t available = blob->range.bytes - offset;
        size_t take = available < remaining ? (size_t)available : remaining;
        if (!read_file(blob->path, offset, take, output)) return 0;
        start += take;
        output += take;
        remaining -= take;
    }
    return remaining == 0 || fail("requested tensor bytes are absent from extracted ranges");
}

static InklingRange tensor_range(const InklingTensor *tensor)
{
    InklingRange range = {tensor->shard_id, tensor->data_offset, tensor->byte_length};
    return range;
}

static const InklingTensor *require_tensor(const InklingModel *model, const char *name,
    InklingDataType dtype, uint32_t rank, uint64_t a, uint64_t b, uint64_t c)
{
    const InklingTensor *tensor = inkling_index_find_tensor(&model->index, name);
    uint64_t dimensions[] = {a, b, c};
    int valid = tensor && tensor->dtype == dtype && tensor->rank == rank && rank <= 3;
    for (uint32_t i = 0; valid && i < rank; i++) valid = tensor->shape[i] == dimensions[i];
    if (!valid) {
        fprintf(stderr, "model: missing tensor or unsupported dtype/shape: %s\n", name);
        return NULL;
    }
    return tensor;
}

static uint64_t little_integer(const unsigned char *bytes, size_t width)
{
    uint64_t result = 0;
    for (size_t i = 0; i < width; i++) result |= (uint64_t)bytes[i] << (8 * i);
    return result;
}

static int floats(const InklingModel *model, const InklingTensor *tensor, float **out)
{
    void *allocation = NULL;
    size_t width = tensor->dtype == INKLING_DTYPE_BF16 ? 2 : 4;
    uint64_t count = tensor->byte_length / width;
    if (!inkling_alloc_array(count, sizeof(float), &allocation)) return fail("cannot allocate tensor vector");
    float *values = allocation;
    void *raw_allocation = NULL;
    if (!inkling_alloc_array(tensor->byte_length, 1, &raw_allocation)) {
        free(values);
        return fail("cannot allocate raw vector");
    }
    unsigned char *raw = raw_allocation;
    if (!inkling_model_read(model, tensor_range(tensor), 0, (size_t)tensor->byte_length, raw)) {
        free(raw);
        free(values);
        return 0;
    }
    for (uint64_t i = 0; i < count; i++) {
        const unsigned char *bytes = raw + i * width;
        if (width == 2) values[i] = inkling_bf16_to_f32((uint16_t)little_integer(bytes, 2));
        else {
            uint32_t bits = (uint32_t)little_integer(bytes, 4);
            memcpy(&values[i], &bits, sizeof(bits));
        }
        if (!isfinite(values[i])) {
            fprintf(stderr, "model: nonfinite vector value in %s\n", tensor->name);
            free(raw);
            free(values);
            return 0;
        }
    }
    free(raw);
    *out = values;
    return 1;
}

static int vector(const InklingModel *model, const char *name, InklingDataType dtype,
                  uint32_t rank, uint64_t a, uint64_t b, uint64_t c, float **out)
{
    const InklingTensor *tensor = require_tensor(model, name, dtype, rank, a, b, c);
    return tensor && floats(model, tensor, out);
}

static int scalar(const InklingModel *model, const char *name, InklingDataType dtype, float *out)
{
    float *value = NULL;
    if (!vector(model, name, dtype, 1, 1, 0, 0, &value)) return 0;
    *out = value[0];
    free(value);
    return 1;
}

static int matrix(const InklingModel *model, const char *name, size_t experts,
                  size_t rows, size_t columns, uint32_t rank, int allow_packed, InklingMatrix *out)
{
    const InklingTensor *tensor = inkling_index_find_tensor(&model->index, name);
    if (!tensor) {
        fprintf(stderr, "model: missing matrix %s\n", name);
        return 0;
    }
    uint64_t a = rank == 3 ? experts : rows, b = rank == 3 ? rows : columns;
    uint64_t c = rank == 3 ? columns : 0;
    out->experts = experts;
    out->rows = rows;
    out->columns = columns;
    if (tensor->dtype == INKLING_DTYPE_BF16) {
        if (!require_tensor(model, name, INKLING_DTYPE_BF16, rank, a, b, c)) return 0;
        out->kind = INKLING_MATRIX_BF16;
        out->data = tensor_range(tensor);
        return 1;
    }
    if (!allow_packed || rank != 3 || tensor->dtype != INKLING_DTYPE_U8 || columns % 16 ||
        !require_tensor(model, name, INKLING_DTYPE_U8, 3, experts, rows, columns / 2))
        return fail("unsupported matrix storage");
    char auxiliary[INKLING_INDEX_MAX_NAME_LENGTH];
    int length = snprintf(auxiliary, sizeof(auxiliary), "%s.original_shape", name);
    if (length < 0 || (size_t)length >= sizeof(auxiliary)) return fail("matrix auxiliary name too long");
    const InklingTensor *original = require_tensor(model, auxiliary, INKLING_DTYPE_I64, 1, 3, 0, 0);
    unsigned char original_bytes[24];
    if (!original || !inkling_model_read(model, tensor_range(original), 0, sizeof(original_bytes), original_bytes)) return 0;
    uint64_t shape[3] = {experts, rows, columns};
    for (size_t i = 0; i < 3; i++) {
        uint64_t stored = little_integer(original_bytes + 8 * i, 8);
        if (stored > INT64_MAX || stored != shape[i]) return fail("NVFP4 original_shape payload mismatch");
    }
    snprintf(auxiliary, sizeof(auxiliary), "%s.scale", name);
    const InklingTensor *scales = require_tensor(model, auxiliary, INKLING_DTYPE_F8_E4M3, 3,
                                                experts, rows, columns / 16);
    snprintf(auxiliary, sizeof(auxiliary), "%s.scale2", name);
    const InklingTensor *scale2 = require_tensor(model, auxiliary, INKLING_DTYPE_F32, 1, experts, 0, 0);
    if (!scales || !scale2 || !inkling_nvfp4_layout(shape, tensor->shape, scales->shape,
                                                  experts, &out->layout) || !floats(model, scale2, &out->scale2)) return 0;
    for (size_t i = 0; i < experts; i++) if (out->scale2[i] <= 0) return fail("NVFP4 scale2 must be positive");
    snprintf(auxiliary, sizeof(auxiliary), "%s.input_amax", name);
    float input_amax;
    if (!scalar(model, auxiliary, INKLING_DTYPE_BF16, &input_amax) || input_amax <= 0)
        return fail("NVFP4 input_amax must be positive");
    out->kind = INKLING_MATRIX_NVFP4;
    out->data = tensor_range(tensor);
    out->scales = tensor_range(scales);
    return 1;
}

static int local_layer(const InklingConfig *config, size_t id)
{
    for (size_t i = 0; i < config->num_local_layers; i++) if (config->local_layer_ids[i] == id) return 1;
    return 0;
}

static int bind_layer(InklingModel *model, InklingLayer *layer, size_t id)
{
    const InklingConfig *config = &model->config;
    int local = local_layer(config, id);
    layer->original_id = id;
    layer->heads = local ? config->swa_num_attention_heads : config->num_attention_heads;
    layer->kv_heads = local ? config->swa_num_key_value_heads : config->num_key_value_heads;
    layer->head_dim = local ? config->swa_head_dim : config->head_dim;
    layer->window = local ? config->sliding_window_size : 0;
    layer->extent = local ? layer->window : config->relative_extent;
    size_t q, kv, r, dense_rows, expert_rows, router_rows;
    if (!inkling_size_mul(layer->heads, layer->head_dim, &q) ||
        !inkling_size_mul(layer->kv_heads, layer->head_dim, &kv) ||
        !inkling_size_mul(layer->heads, config->relative_dimension, &r) ||
        !inkling_size_mul(config->dense_intermediate_size, 2, &dense_rows) ||
        !inkling_size_mul(config->expert_intermediate_size, 2, &expert_rows) ||
        !inkling_size_add(config->num_routed_experts, config->num_shared_experts, &router_rows))
        return fail("layer geometry overflow");
    size_t h = config->hidden_size;
    char name[INKLING_INDEX_MAX_NAME_LENGTH];
#define NAME(suffix) snprintf(name, sizeof(name), "model.llm.layers.%zu.%s", id, suffix)
#define MAT(suffix, member, experts_, rows_, cols_, rank_, packed_) \
    do { NAME(suffix); if (!matrix(model, name, experts_, rows_, cols_, rank_, packed_, &layer->member)) return 0; } while (0)
#define VEC(suffix, member, rank_, a_, b_, c_) \
    do { NAME(suffix); if (!vector(model, name, INKLING_DTYPE_BF16, rank_, a_, b_, c_, &layer->member)) return 0; } while (0)
    MAT("attn.wq_du.weight", q, 1, q, h, 2, 0);
    MAT("attn.wk_dv.weight", k, 1, kv, h, 2, 0);
    MAT("attn.wv_dv.weight", v, 1, kv, h, 2, 0);
    MAT("attn.wr_du.weight", r, 1, r, h, 2, 0);
    MAT("attn.wo_ud.weight", o, 1, h, q, 2, 0);
    VEC("attn.q_norm.weight", q_norm, 1, layer->head_dim, 0, 0);
    VEC("attn.k_norm.weight", k_norm, 1, layer->head_dim, 0, 0);
    VEC("attn.rel_logits_proj.proj", relative, 2, config->relative_dimension, layer->extent, 0);
    VEC("attn.k_sconv.weight", k_conv, 3, kv, 1, 4);
    VEC("attn.v_sconv.weight", v_conv, 3, kv, 1, 4);
    VEC("attn_norm.weight", attn_norm, 1, h, 0, 0);
    VEC("mlp_norm.weight", mlp_norm, 1, h, 0, 0);
    VEC("attn_sconv.weight", attn_conv, 3, h, 1, 4);
    VEC("mlp_sconv.weight", mlp_conv, 3, h, 1, 4);
    layer->dense = id < config->dense_mlp_idx;
    if (layer->dense) {
        MAT("mlp.w13_dn.weight", w13, 1, dense_rows, h, 2, 0);
        MAT("mlp.w2_md.weight", down, 1, h, config->dense_intermediate_size, 2, 0);
        NAME("mlp.global_scale");
        return scalar(model, name, INKLING_DTYPE_BF16, &layer->global_scale);
    }
    MAT("mlp.experts.w13_weight", w13, config->num_routed_experts, expert_rows, h, 3, 1);
    MAT("mlp.experts.w2_weight", down, config->num_routed_experts, h, config->expert_intermediate_size, 3, 1);
    MAT("mlp.shared_experts.shared_w13_weight", shared_w13, config->num_shared_experts, expert_rows, h, 3, 0);
    MAT("mlp.shared_experts.shared_w2_weight", shared_down, config->num_shared_experts, h, config->expert_intermediate_size, 3, 0);
    MAT("mlp.gate.weight", router, 1, router_rows, h, 2, 0);
    NAME("mlp.gate.bias");
    if (!vector(model, name, INKLING_DTYPE_F32, 1, config->num_routed_experts, 0, 0, &layer->correction)) return 0;
    NAME("mlp.gate.global_scale");
    return scalar(model, name, INKLING_DTYPE_F32, &layer->global_scale);
#undef NAME
#undef MAT
#undef VEC
}

static void free_matrix(InklingMatrix *matrix_value)
{
    free(matrix_value->scale2);
}

void inkling_model_close(InklingModel *model)
{
    if (!model) return;
    for (size_t i = 0; i < model->layer_count; i++) {
        InklingLayer *layer = &model->layers[i];
        free(layer->q_norm); free(layer->k_norm); free(layer->relative);
        free(layer->k_conv); free(layer->v_conv);
        free(layer->attn_norm); free(layer->mlp_norm); free(layer->attn_conv); free(layer->mlp_conv);
        free(layer->correction);
        free_matrix(&layer->q); free_matrix(&layer->k); free_matrix(&layer->v);
        free_matrix(&layer->r); free_matrix(&layer->o);
        free_matrix(&layer->w13); free_matrix(&layer->down);
        free_matrix(&layer->shared_w13); free_matrix(&layer->shared_down); free_matrix(&layer->router);
    }
    for (size_t i = 0; i < model->shard_count; i++) free(model->shards[i].path);
    for (size_t i = 0; i < model->blob_count; i++) free(model->blobs[i].path);
    free(model->shards); free(model->blobs); free(model->layers);
    free(model->embed_norm); free(model->final_norm);
    free_matrix(&model->embed); free_matrix(&model->unembed);
    inkling_packed_close(model->packed);
    inkling_config_free(&model->config);
    inkling_index_free(&model->index);
    free(model);
}

size_t inkling_model_logits_width(const InklingModel *model)
{
    return model && model->complete ? model->config.unpadded_vocab_size : 0;
}

static int load_shards(InklingModel *model, const char *directory, int extracted)
{
    void *allocation = NULL;
    if (!inkling_alloc_array(model->index.num_shards, sizeof(*model->shards), &allocation)) return 0;
    model->shards = allocation;
    model->shard_count = model->index.num_shards;
    memset(model->shards, 0, model->shard_count * sizeof(*model->shards));
    for (uint64_t i = 0; i < model->index.count; i++) {
        const InklingTensor *tensor = &model->index.entries[i];
        InklingShard *shard = &model->shards[tensor->shard_id];
        if (shard->path || (!extracted && strcmp(tensor->shard, "mtp.safetensors") == 0)) continue;
        char *base = extracted ? join(directory, "headers") : NULL;
        shard->path = join(extracted ? base : directory, tensor->shard);
        free(base);
        if (!shard->path) return fail("cannot allocate shard path");
        char *header = NULL;
        if (!inkling_safetensors_read_header(shard->path, &header, &shard->header_size)) return fail("cannot read shard header");
        int ok;
        if (extracted) {
            InklingJsonDocument document = {0};
            ok = inkling_json_parse(header, (size_t)shard->header_size, &document, NULL);
            uint64_t end = 0;
            for (size_t j = 0; ok && j < inkling_json_object_size(document.root); j++) {
                const InklingJsonValue *entry = inkling_json_object_value_at(document.root, j);
                const InklingJsonValue *offsets = inkling_json_object_get(entry, "data_offsets");
                if (!offsets) continue;
                uint64_t value;
                ok = inkling_json_number_u64(inkling_json_array_at(offsets, 1), &value);
                if (ok && value > end) end = value;
            }
            inkling_json_document_free(&document);
            if (end > UINT64_MAX - 8 - shard->header_size) ok = 0;
            if (ok) {
                shard->file_size = 8 + shard->header_size + end;
                ok = inkling_index_bind_header(&model->index, tensor->shard, header,
                                               shard->header_size, shard->file_size);
            }
        } else {
            ok = file_size(shard->path, &shard->file_size) &&
                 inkling_index_bind_header(&model->index, tensor->shard, header,
                                           shard->header_size, shard->file_size);
        }
        free(header);
        if (!ok) return fail("invalid shard metadata");
    }
    uint64_t total = 0;
    int all_bound = 1;
    for (uint64_t i = 0; i < model->index.count; i++) {
        const InklingTensor *tensor = &model->index.entries[i];
        if (tensor->dtype == INKLING_DTYPE_UNKNOWN) { all_bound = 0; continue; }
        if (total > UINT64_MAX - tensor->byte_length) return fail("catalogue byte total overflow");
        total += tensor->byte_length;
    }
    if (all_bound && total != model->index.total_size) return fail("index total_size differs from bound shard payloads");
    return 1;
}

static int string_is(const InklingJsonValue *value, const char *expected)
{
    size_t length;
    const char *text = inkling_json_string(value, &length);
    return text && length == strlen(expected) && memcmp(text, expected, length) == 0;
}

static char *blob_path(const char *directory, const InklingJsonValue *value)
{
    size_t length;
    const char *relative = inkling_json_string(value, &length);
    if (!relative || !length || strlen(relative) != length || relative[0] == '/' || strchr(relative, '\\')) return NULL;
    const char *part = relative;
    for (size_t i = 0; i <= length; i++) {
        if (relative[i] != '/' && relative[i] != '\0') continue;
        size_t size = (size_t)(relative + i - part);
        if (!size || (size == 1 && part[0] == '.') || (size == 2 && part[0] == '.' && part[1] == '.')) return NULL;
        part = relative + i + 1;
    }
    char *parent = realpath(directory, NULL), *joined = join(directory, relative);
    char *resolved = joined ? realpath(joined, NULL) : NULL;
    free(joined);
    if (!parent || !resolved || strncmp(parent, resolved, strlen(parent)) != 0 ||
        (strcmp(parent, "/") != 0 && resolved[strlen(parent)] != '/')) {
        free(resolved);
        resolved = NULL;
    }
    free(parent);
    return resolved;
}

static int compare_blobs(const void *left, const void *right)
{
    const InklingBlob *a = left, *b = right;
    if (a->range.shard != b->range.shard) return a->range.shard < b->range.shard ? -1 : 1;
    return a->range.start < b->range.start ? -1 : a->range.start != b->range.start;
}

static int load_manifest(InklingModel *model, const char *path, size_t *layer)
{
    InklingJsonDocument document = {0};
    int ok = 0;
    if (!inkling_json_load_file(path, 64 * 1024 * 1024, &document)) return 0;
    uint64_t schema, id;
    const InklingJsonValue *root = document.root;
    if (!inkling_json_number_u64(inkling_json_object_get(root, "schema"), &schema) || schema != 1 ||
        !string_is(inkling_json_object_get(root, "checkpoint_revision"), REVISION) ||
        !inkling_json_number_u64(inkling_json_object_get(root, "layer"), &id) || id >= model->config.num_hidden_layers)
        goto done;
    *layer = (size_t)id;
    char *directory = strdup(path);
    if (!directory) goto done;
    char *slash = strrchr(directory, '/');
    if (slash) { if (slash == directory) slash[1] = '\0'; else *slash = '\0'; }
    else { free(directory); directory = strdup("."); }
    if (!directory) goto done;
    const InklingJsonValue *ranges = inkling_json_object_get(root, "ranges");
    size_t count = inkling_json_array_size(ranges);
    void *allocation = NULL;
    if (inkling_json_type(ranges) != INKLING_JSON_ARRAY || !count ||
        !inkling_alloc_array(count, sizeof(*model->blobs), &allocation)) { free(directory); goto done; }
    model->blobs = allocation;
    model->blob_count = count;
    memset(model->blobs, 0, count * sizeof(*model->blobs));
    ok = 1;
    for (size_t i = 0; ok && i < count; i++) {
        const InklingJsonValue *entry = inkling_json_array_at(ranges, i);
        size_t length;
        const char *name = inkling_json_string(inkling_json_object_get(entry, "tensor"), &length);
        uint64_t offset, bytes, actual;
        const InklingTensor *tensor = name && strlen(name) == length ? inkling_index_find_tensor(&model->index, name) : NULL;
        if (!tensor ||
            !inkling_json_number_u64(inkling_json_object_get(entry, "offset"), &offset) ||
            !inkling_json_number_u64(inkling_json_object_get(entry, "bytes"), &bytes) || !bytes ||
            offset > tensor->byte_length || bytes > tensor->byte_length - offset ||
            tensor->data_offset > UINT64_MAX - offset) { ok = 0; break; }
        const char *hash = inkling_json_string(inkling_json_object_get(entry, "sha256"), &length);
        if (!hash || length != 64 || strlen(hash) != length) { ok = 0; break; }
        for (size_t j = 0; j < length; j++)
            if (!((hash[j] >= '0' && hash[j] <= '9') || (hash[j] >= 'a' && hash[j] <= 'f') ||
                  (hash[j] >= 'A' && hash[j] <= 'F'))) ok = 0;
        InklingBlob *blob = &model->blobs[i];
        blob->path = blob_path(directory, inkling_json_object_get(entry, "file"));
        blob->range.shard = tensor->shard_id;
        blob->range.start = tensor->data_offset + offset;
        blob->range.bytes = bytes;
        if (!ok || !blob->path || !file_size(blob->path, &actual) || actual != bytes) { ok = 0; break; }
    }
    free(directory);
    if (!ok) goto done;
    qsort(model->blobs, count, sizeof(*model->blobs), compare_blobs);
    for (size_t i = 1; i < count; i++) {
        InklingRange previous = model->blobs[i - 1].range, current = model->blobs[i].range;
        if (previous.shard == current.shard && current.start - previous.start < previous.bytes) { ok = 0; break; }
    }
done:
    inkling_json_document_free(&document);
    return ok || fail("invalid extracted-range manifest (identity, tensor bounds, paths, lengths or overlap)");
}

static int integer_is(const InklingJsonValue *value, uint64_t expected)
{
    uint64_t number;
    return inkling_json_number_u64(value, &number) && number == expected;
}

static int pair_is(const InklingJsonValue *value, uint64_t first, uint64_t second)
{
    return inkling_json_type(value) == INKLING_JSON_ARRAY && inkling_json_array_size(value) == 2 &&
        integer_is(inkling_json_array_at(value, 0), first) && integer_is(inkling_json_array_at(value, 1), second);
}

static int flag_is(const InklingJsonValue *object, const char *key, int expected)
{
    int flag;
    return inkling_json_boolean(inkling_json_object_get(object, key), &flag) && flag == expected;
}

static int matrix_name(const char *name)
{
    static const char *suffixes[] = {
        "embed.weight", "unembed.weight", "attn.wq_du.weight", "attn.wk_dv.weight",
        "attn.wv_dv.weight", "attn.wr_du.weight", "attn.wo_ud.weight",
        "mlp.w13_dn.weight", "mlp.w2_md.weight", "mlp.experts.w13_weight", "mlp.experts.w2_weight",
        "mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.shared_w2_weight", "mlp.gate.weight"
    };
    size_t length = strlen(name);
    if (strncmp(name, "model.llm.", 10) != 0) return 0;
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        size_t suffix_length = strlen(suffixes[i]);
        if (length > suffix_length && name[length - suffix_length - 1] == '.' &&
            strcmp(name + length - suffix_length, suffixes[i]) == 0) return 1;
    }
    return 0;
}

static int excluded(const InklingJsonValue *modules, const char *name)
{
    for (size_t i = 0; i < inkling_json_array_size(modules); i++) {
        size_t length;
        const char *prefix = inkling_json_string(inkling_json_array_at(modules, i), &length);
        if (strlen(name) >= length && memcmp(name, prefix, length) == 0 &&
            (name[length] == '\0' || name[length] == '.')) return 1;
    }
    return 0;
}

static int quantization(const InklingModel *model, const char *directory)
{
    char *path = join(directory, "hf_quant_config.json");
    InklingJsonDocument document = {0};
    int ok = path && inkling_json_load_file(path, 16 * 1024 * 1024, &document);
    free(path);
    if (!ok) return fail("cannot load text quantization metadata");
    const InklingJsonValue *quant = inkling_json_object_get(document.root, "quantization");
    const InklingJsonValue *modules = inkling_json_object_get(quant, "exclude_modules");
    const InklingJsonValue *modelopt = inkling_json_object_get(quant, "modelopt_quant_config");
    const InklingJsonValue *settings = inkling_json_object_get(modelopt, "quant_cfg");
    ok = string_is(inkling_json_object_get(quant, "quant_algo"), "NVFP4") &&
        string_is(inkling_json_object_get(quant, "kv_cache_quant_algo"), "none") &&
        integer_is(inkling_json_object_get(quant, "group_size"), 16) &&
        inkling_json_type(modules) == INKLING_JSON_ARRAY &&
        string_is(inkling_json_object_get(modelopt, "algorithm"), "max") &&
        flag_is(inkling_json_object_get(settings, "default"), "enable", 0);
    const char *keys[] = {"*weight_quantizer", "*input_quantizer"};
    for (size_t i = 0; ok && i < 2; i++) {
        const InklingJsonValue *entry = inkling_json_object_get(settings, keys[i]);
        const InklingJsonValue *block = inkling_json_object_get(entry, "block_sizes");
        const InklingJsonValue *axis = inkling_json_object_get(entry, "axis");
        ok = pair_is(inkling_json_object_get(entry, "num_bits"), 2, 1) && flag_is(entry, "enable", 1) &&
            axis && inkling_json_type(axis) == INKLING_JSON_NULL &&
            inkling_json_object_size(block) == 3 && integer_is(inkling_json_object_get(block, "-1"), 16) &&
            string_is(inkling_json_object_get(block, "type"), "dynamic") &&
            pair_is(inkling_json_object_get(block, "scale_bits"), 4, 3);
    }
    for (size_t i = 0; ok && i < inkling_json_array_size(modules); i++) {
        size_t length;
        const char *prefix = inkling_json_string(inkling_json_array_at(modules, i), &length);
        ok = prefix && length && strlen(prefix) == length;
    }
    for (uint64_t i = 0; ok && i < model->index.count; i++) {
        const InklingTensor *tensor = &model->index.entries[i];
        if (!matrix_name(tensor->name)) continue;
        int unquantized = excluded(modules, tensor->name);
        ok = (tensor->dtype == INKLING_DTYPE_BF16 && unquantized) ||
             (tensor->dtype == INKLING_DTYPE_U8 && !unquantized);
        if (!ok) fprintf(stderr, "model: quantization exclusion disagrees with stored matrix %s\n", tensor->name);
    }
    inkling_json_document_free(&document);
    return ok || fail("unsupported text quantization metadata (expected NVFP4, group 16, matching exclusions)");
}

static int open_model(const char *directory, const char *manifest, int production,
                      const InklingModelOptions *options, InklingModel **out)
{
    if (!out) return fail("model output pointer is NULL");
    *out = NULL;
    if (!directory) return fail("model directory is NULL");
    if ((production || manifest) && !inkling_verify_model_quiet(directory, manifest != NULL)) return 0;
    InklingModel *model = calloc(1, sizeof(*model));
    if (!model) return fail("cannot allocate model");
    model->extracted = manifest != NULL;
    char *path = join(directory, "config.json");
    int ok = path && inkling_config_load(path, &model->config);
    free(path);
    if (!ok) { fail("cannot load a valid text configuration"); goto done; }
    if (!model->config.execution_fields_valid) { fail("missing or unsupported text execution configuration"); ok = 0; goto done; }
    path = join(directory, "model.safetensors.index.json");
    ok = path && inkling_index_load(path, &model->index);
    free(path);
    if (!ok || !load_shards(model, directory, model->extracted) || !quantization(model, directory)) { ok = 0; goto done; }
    if (options && options->packed_trunk_path &&
        !inkling_packed_open(model, directory, options->packed_trunk_path)) { ok = 0; goto done; }
    size_t layer = 0;
    if (manifest && !load_manifest(model, manifest, &layer)) { ok = 0; goto done; }
    size_t layer_count = manifest ? 1 : model->config.num_hidden_layers;
    void *allocation = NULL;
    if (!inkling_alloc_array(layer_count, sizeof(*model->layers), &allocation)) { fail("cannot allocate layers"); ok = 0; goto done; }
    model->layers = allocation;
    model->layer_count = layer_count;
    memset(model->layers, 0, model->layer_count * sizeof(*model->layers));
    for (size_t i = 0; i < model->layer_count; i++)
        if (!bind_layer(model, &model->layers[i], manifest ? layer : i)) { ok = 0; goto done; }
    if (!manifest) {
        size_t h = model->config.hidden_size, vocab = model->config.vocab_size;
        if (!matrix(model, "model.llm.embed.weight", 1, vocab, h, 2, 0, &model->embed) ||
            !matrix(model, "model.llm.unembed.weight", 1, vocab, h, 2, 0, &model->unembed) ||
            !vector(model, "model.llm.embed_norm.weight", INKLING_DTYPE_BF16, 1, h, 0, 0, &model->embed_norm) ||
            !vector(model, "model.llm.norm.weight", INKLING_DTYPE_BF16, 1, h, 0, 0, &model->final_norm)) { ok = 0; goto done; }
        model->complete = 1;
    }
    *out = model;
    return 1;
done:
    inkling_model_close(model);
    return ok ? fail("cannot bind model") : 0;
}

int inkling_model_open(const char *directory, InklingModel **out)
{
    return inkling_model_open_with_options(directory, NULL, out);
}

int inkling_model_open_with_options(const char *directory, const InklingModelOptions *options, InklingModel **out)
{
    return open_model(directory, NULL, 1, options, out);
}

int inkling_model_open_text_fixture(const char *directory, InklingModel **out)
{
    return inkling_model_open_text_fixture_with_options(directory, NULL, out);
}

int inkling_model_open_text_fixture_with_options(const char *directory,
    const InklingModelOptions *options, InklingModel **out)
{
    return open_model(directory, NULL, 0, options, out);
}

int inkling_model_open_layer_fixture(const char *directory, const char *manifest, InklingModel **out)
{
    if (!manifest) { if (out) *out = NULL; return fail("layer fixture requires a manifest"); }
    return open_model(directory, manifest, 0, NULL, out);
}

size_t inkling_model_max_length(const InklingModel *model)
{
    return model != NULL && model->complete ? model->config.model_max_length : 0;
}
