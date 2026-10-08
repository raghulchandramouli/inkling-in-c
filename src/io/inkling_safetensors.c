#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"
#include "inkling_json.h"

#define SAFETENSORS_MAX_HEADER_SIZE (100ULL * 1024ULL * 1024ULL)

static int read_header_size(FILE *file, uint64_t *size)
{
    unsigned char prefix[8];
    if (fread(prefix, 1, sizeof(prefix), file) != sizeof(prefix)) {
        return 0;
    }
    uint64_t length = 0;
    for (unsigned int byte = 0; byte < sizeof(prefix); byte++) {
        length |= (uint64_t)prefix[byte] << (byte * 8);
    }
    if (length == 0 || length > SAFETENSORS_MAX_HEADER_SIZE || length >= SIZE_MAX) {
        return 0;
    }
    *size = length;
    return 1;
}

static int file_length(FILE *file, uint64_t *length)
{
    if (fseek(file, 0, SEEK_END) != 0) {
        return 0;
    }
    long size = ftell(file);
    if (size < 0) {
        return 0;
    }
    *length = (uint64_t)size;
    return 1;
}

static int read_file_header(const char *path, char **json, uint64_t *size, uint64_t *file_size)
{
    if (path == NULL || json == NULL || size == NULL) {
        return 0;
    }
    *json = NULL;
    *size = 0;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    uint64_t length = 0;
    int success = read_header_size(file, &length);
    char *text = success ? malloc((size_t)length + 1) : NULL;
    success = text != NULL && fread(text, 1, (size_t)length, file) == (size_t)length;
    if (success && file_size != NULL) {
        success = file_length(file, file_size);
    }
    fclose(file);
    if (!success) {
        free(text);
        return 0;
    }
    text[length] = '\0';
    *json = text;
    *size = length;
    return 1;
}

int inkling_safetensors_read_header(const char *path, char **json, uint64_t *size)
{
    return read_file_header(path, json, size, NULL);
}

static uint64_t dtype_size(InklingDataType dtype)
{
    switch (dtype) {
    case INKLING_DTYPE_F32: return 4;
    case INKLING_DTYPE_BF16: return 2;
    case INKLING_DTYPE_I64: return 8;
    case INKLING_DTYPE_F8_E4M3:
    case INKLING_DTYPE_U8: return 1;
    default: return 0;
    }
}

static int tensor_size_is_valid(const InklingTensorInfo *tensor)
{
    uint64_t width = dtype_size(tensor->dtype);
    if (width == 0 || tensor->rank > INKLING_MAX_TENSOR_RANK ||
        tensor->data_end < tensor->data_start) {
        return 0;
    }
    uint64_t elements = 1;
    for (uint32_t dimension = 0; dimension < tensor->rank; dimension++) {
        uint64_t size = tensor->shape[dimension];
        if (size != 0 && elements > UINT64_MAX / size) {
            return 0;
        }
        elements *= size;
    }
    return elements <= UINT64_MAX / width &&
           elements * width == tensor->data_end - tensor->data_start;
}

static int parse_tensor(const InklingJsonValue *value, InklingTensorInfo *tensor)
{
    if (inkling_json_type(value) != INKLING_JSON_OBJECT || inkling_json_object_size(value) != 3) {
        return 0;
    }
    size_t length = 0;
    const char *dtype = inkling_json_string(inkling_json_object_get(value, "dtype"), &length);
    static const struct { const char *name; InklingDataType dtype; } types[] = {
        {"F32", INKLING_DTYPE_F32}, {"BF16", INKLING_DTYPE_BF16},
        {"I64", INKLING_DTYPE_I64}, {"F8_E4M3", INKLING_DTYPE_F8_E4M3},
        {"U8", INKLING_DTYPE_U8}
    };
    for (size_t type = 0; dtype != NULL && type < sizeof(types) / sizeof(types[0]); type++) {
        if (length == strlen(types[type].name) && memcmp(dtype, types[type].name, length) == 0) {
            tensor->dtype = types[type].dtype;
            break;
        }
    }
    const InklingJsonValue *shape = inkling_json_object_get(value, "shape");
    size_t rank = inkling_json_array_size(shape);
    if (tensor->dtype == INKLING_DTYPE_UNKNOWN || inkling_json_type(shape) != INKLING_JSON_ARRAY ||
        rank > INKLING_MAX_TENSOR_RANK) {
        return 0;
    }
    tensor->rank = (uint32_t)rank;
    for (size_t dimension = 0; dimension < rank; dimension++) {
        if (!inkling_json_number_u64(inkling_json_array_at(shape, dimension), &tensor->shape[dimension])) {
            return 0;
        }
    }
    const InklingJsonValue *offsets = inkling_json_object_get(value, "data_offsets");
    return inkling_json_type(offsets) == INKLING_JSON_ARRAY && inkling_json_array_size(offsets) == 2 &&
           inkling_json_number_u64(inkling_json_array_at(offsets, 0), &tensor->data_start) &&
           inkling_json_number_u64(inkling_json_array_at(offsets, 1), &tensor->data_end) &&
           tensor_size_is_valid(tensor);
}

static int compare_offsets(const void *left, const void *right)
{
    const InklingTensor *a = left, *b = right;
    if (a->data_offset != b->data_offset) {
        return a->data_offset < b->data_offset ? -1 : 1;
    }
    return a->byte_length < b->byte_length ? -1 : a->byte_length > b->byte_length;
}

int inkling_safetensors_parse_header(const char *json, uint64_t header_size, uint64_t shard_size,
                                    InklingTensor **tensors, size_t *count)
{
    if (tensors == NULL || count == NULL) {
        return 0;
    }
    *tensors = NULL;
    *count = 0;
    if (json == NULL || header_size == 0 || header_size > SAFETENSORS_MAX_HEADER_SIZE ||
        header_size >= SIZE_MAX || shard_size < 8 || header_size > shard_size - 8 || json[0] != '{') {
        return 0;
    }
    InklingJsonDocument document = {0};
    if (!inkling_json_parse(json, (size_t)header_size, &document, NULL)) {
        return 0;
    }
    size_t members = inkling_json_object_size(document.root);
    int success = inkling_json_type(document.root) == INKLING_JSON_OBJECT &&
                  members <= SIZE_MAX / sizeof(InklingTensor);
    InklingTensor *parsed = success && members != 0 ? calloc(members, sizeof(*parsed)) : NULL;
    if (members != 0 && parsed == NULL) {
        success = 0;
    }
    size_t used = 0;
    uint64_t payload_size = shard_size - 8 - header_size;
    for (size_t member = 0; success && member < members; member++) {
        size_t length = 0;
        const char *name = inkling_json_object_key_at(document.root, member, &length);
        const InklingJsonValue *value = inkling_json_object_value_at(document.root, member);
        if (length == 12 && memcmp(name, "__metadata__", 12) == 0) {
            success = inkling_json_type(value) == INKLING_JSON_OBJECT;
            for (size_t item = 0; success && item < inkling_json_object_size(value); item++) {
                success = inkling_json_string(inkling_json_object_value_at(value, item), NULL) != NULL;
            }
            continue;
        }
        InklingTensorInfo info = {0};
        if (length == 0 || length >= sizeof(parsed[used].name) || memchr(name, '\0', length) != NULL ||
            !parse_tensor(value, &info) || info.data_end > payload_size) {
            success = 0;
            break;
        }
        InklingTensor *tensor = &parsed[used++];
        memcpy(tensor->name, name, length + 1);
        tensor->dtype = info.dtype;
        tensor->rank = info.rank;
        memcpy(tensor->shape, info.shape, sizeof(tensor->shape));
        tensor->data_offset = info.data_start;
        tensor->byte_length = info.data_end - info.data_start;
    }
    inkling_json_document_free(&document);
    if (success) {
        if (used != 0) qsort(parsed, used, sizeof(*parsed), compare_offsets);
        uint64_t end = 0;
        for (size_t position = 0; position < used; position++) {
            if (parsed[position].data_offset != end) {
                success = 0;
                break;
            }
            end = parsed[position].data_offset + parsed[position].byte_length;
        }
        if (end != payload_size) success = 0;
    }
    if (!success) {
        free(parsed);
        return 0;
    }
    *tensors = parsed;
    *count = used;
    return 1;
}

int inkling_safetensors_load_shard(const char *path, InklingTensor **tensors, size_t *count)
{
    if (tensors == NULL || count == NULL) {
        return 0;
    }
    *tensors = NULL;
    *count = 0;
    char *json = NULL;
    uint64_t header_size = 0, file_size = 0;
    if (!read_file_header(path, &json, &header_size, &file_size)) {
        return 0;
    }
    int success = inkling_safetensors_parse_header(json, header_size, file_size, tensors, count);
    free(json);
    return success;
}

int inkling_safetensors_find_tensor(const char *json, const char *name, InklingTensorInfo *tensor)
{
    if (json == NULL || name == NULL || tensor == NULL) {
        return 0;
    }
    InklingJsonDocument document = {0};
    if (!inkling_json_parse(json, strlen(json), &document, NULL)) {
        return 0;
    }
    InklingTensorInfo parsed = {0};
    int success = parse_tensor(inkling_json_object_get(document.root, name), &parsed);
    inkling_json_document_free(&document);
    if (success) {
        *tensor = parsed;
    }
    return success;
}

int inkling_safetensors_read_tensor_data(const char *path, uint64_t header_size,
                                        const InklingTensorInfo *tensor, void *destination,
                                        size_t destination_size)
{
    if (path == NULL || tensor == NULL || destination == NULL || !tensor_size_is_valid(tensor)) {
        return 0;
    }
    uint64_t bytes = tensor->data_end - tensor->data_start;
    if (bytes > destination_size || header_size > UINT64_MAX - 8 ||
        tensor->data_start > UINT64_MAX - 8 - header_size) {
        return 0;
    }
    uint64_t offset = 8 + header_size + tensor->data_start;
    if (offset > LONG_MAX) {
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    uint64_t actual_header = 0, file_size = 0;
    int success = read_header_size(file, &actual_header) && actual_header == header_size &&
                  file_length(file, &file_size) && file_size >= 8 + header_size &&
                  tensor->data_end <= file_size - 8 - header_size &&
                  fseek(file, (long)offset, SEEK_SET) == 0 &&
                  fread(destination, 1, (size_t)bytes, file) == (size_t)bytes;
    fclose(file);
    return success;
}
