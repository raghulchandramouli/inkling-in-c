#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inkling/inkling.h"
#include "inkling_json.h"

#define INDEX_MAX_BYTES (16L * 1024L * 1024L)

static char *read_text_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    long length = -1;
    if (fseek(file, 0, SEEK_END) == 0) {
        length = ftell(file);
    }
    if (length < 0 || length > INDEX_MAX_BYTES || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char *text = malloc((size_t)length + 1);
    if (text == NULL || fread(text, 1, (size_t)length, file) != (size_t)length) {
        free(text);
        fclose(file);
        return NULL;
    }
    fclose(file);
    text[length] = '\0';
    *size = (size_t)length;
    return text;
}

static uint64_t name_hash(const char *name)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (; *name != '\0'; name++) {
        hash ^= (unsigned char)*name;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static size_t find_slot(const InklingIndex *index, const char *name)
{
    size_t slot = (size_t)(name_hash(name) & (uint64_t)(index->slot_count - 1));
    for (size_t probe = 0; probe < index->slot_count; probe++) {
        uint64_t entry = index->slots[slot];
        if (entry == 0 || strcmp(index->entries[entry - 1].name, name) == 0) {
            return slot;
        }
        slot = (slot + 1) & (index->slot_count - 1);
    }
    return index->slot_count;
}

int inkling_index_load(const char *path, InklingIndex *index)
{
    if (path == NULL || index == NULL) {
        return 0;
    }
    size_t length = 0;
    char *json = read_text_file(path, &length);
    if (json == NULL) {
        return 0;
    }
    InklingJsonDocument document = {0};
    int success = inkling_json_parse(json, length, &document, NULL);
    free(json);
    InklingIndex parsed = {0};
    const InklingJsonValue *map = inkling_json_object_get(document.root, "weight_map");
    const InklingJsonValue *metadata = inkling_json_object_get(document.root, "metadata");
    size_t count = inkling_json_object_size(map);
    if (!success || inkling_json_type(map) != INKLING_JSON_OBJECT || count == 0 ||
        count > SIZE_MAX / sizeof(*parsed.entries) || count > SIZE_MAX / 2 ||
        !inkling_json_number_u64(inkling_json_object_get(metadata, "total_size"),
                                 &parsed.total_size)) {
        success = 0;
        goto done;
    }
    parsed.slot_count = 2;
    while (parsed.slot_count < count * 2) {
        if (parsed.slot_count > SIZE_MAX / 2) {
            success = 0;
            goto done;
        }
        parsed.slot_count *= 2;
    }
    if (parsed.slot_count > SIZE_MAX / sizeof(*parsed.slots)) {
        success = 0;
        goto done;
    }
    parsed.entries = calloc(count, sizeof(*parsed.entries));
    parsed.slots = calloc(parsed.slot_count, sizeof(*parsed.slots));
    if (parsed.entries == NULL || parsed.slots == NULL) {
        success = 0;
        goto done;
    }
    parsed.capacity = (uint64_t)count;
    for (size_t position = 0; position < count; position++) {
        size_t name_length = 0, shard_length = 0;
        const char *name = inkling_json_object_key_at(map, position, &name_length);
        const char *shard = inkling_json_string(
            inkling_json_object_value_at(map, position), &shard_length);
        InklingTensor *tensor = &parsed.entries[position];
        if (name_length == 0 || name_length >= sizeof(tensor->name) ||
            memchr(name, '\0', name_length) != NULL || shard == NULL ||
            shard_length == 0 || shard_length >= sizeof(tensor->shard) ||
            memchr(shard, '\0', shard_length) != NULL ||
            strchr(shard, '/') != NULL || strchr(shard, '\\') != NULL ||
            strcmp(shard, ".") == 0 || strcmp(shard, "..") == 0) {
            success = 0;
            goto done;
        }
        memcpy(tensor->name, name, name_length + 1);
        memcpy(tensor->shard, shard, shard_length + 1);
        size_t previous = 0;
        for (; previous < position; previous++) {
            if (strcmp(parsed.entries[previous].shard, shard) == 0) {
                tensor->shard_id = parsed.entries[previous].shard_id;
                break;
            }
        }
        if (previous == position) {
            if (parsed.num_shards == UINT32_MAX) {
                success = 0;
                goto done;
            }
            tensor->shard_id = parsed.num_shards++;
        }
        size_t slot = find_slot(&parsed, name);
        if (slot == parsed.slot_count || parsed.slots[slot] != 0) {
            success = 0;
            goto done;
        }
        parsed.slots[slot] = (uint64_t)position + 1;
        parsed.count++;
    }
done:
    inkling_json_document_free(&document);
    if (!success) {
        inkling_index_free(&parsed);
        return 0;
    }
    *index = parsed;
    return 1;
}

const InklingTensor *inkling_index_find_tensor(const InklingIndex *index, const char *name)
{
    if (index == NULL || name == NULL || index->slots == NULL || index->slot_count == 0) {
        return NULL;
    }
    size_t slot = find_slot(index, name);
    if (slot == index->slot_count || index->slots[slot] == 0) {
        return NULL;
    }
    return &index->entries[index->slots[slot] - 1];
}

int inkling_index_find_shard(const InklingIndex *index, const char *name,
                             char *output, size_t output_size)
{
    const InklingTensor *tensor = inkling_index_find_tensor(index, name);
    if (tensor == NULL || output == NULL || strlen(tensor->shard) >= output_size) {
        return 0;
    }
    memcpy(output, tensor->shard, strlen(tensor->shard) + 1);
    return 1;
}

static int bind_tensors(InklingIndex *index, const char *shard,
                        const InklingTensor *tensors, size_t count)
{
    size_t expected = 0;
    for (uint64_t entry = 0; entry < index->count; entry++) {
        expected += strcmp(index->entries[entry].shard, shard) == 0 ? 1U : 0U;
    }
    if (expected == 0 || count != expected) {
        return 0;
    }
    /* Check the entire shard before publishing any metadata. */
    for (size_t position = 0; position < count; position++) {
        const InklingTensor *entry = inkling_index_find_tensor(index, tensors[position].name);
        if (entry == NULL || strcmp(entry->shard, shard) != 0) {
            return 0;
        }
    }
    for (size_t position = 0; position < count; position++) {
        size_t slot = find_slot(index, tensors[position].name);
        InklingTensor *entry = &index->entries[index->slots[slot] - 1];
        entry->dtype = tensors[position].dtype;
        entry->rank = tensors[position].rank;
        memcpy(entry->shape, tensors[position].shape, sizeof(entry->shape));
        entry->data_offset = tensors[position].data_offset;
        entry->byte_length = tensors[position].byte_length;
    }
    return 1;
}

int inkling_index_bind_header(InklingIndex *index, const char *shard, const char *json,
                              uint64_t header_size, uint64_t shard_size)
{
    if (index == NULL || shard == NULL) {
        return 0;
    }
    InklingTensor *tensors = NULL;
    size_t count = 0;
    int success = inkling_safetensors_parse_header(json, header_size, shard_size, &tensors, &count);
    if (success) {
        success = bind_tensors(index, shard, tensors, count);
    }
    free(tensors);
    return success;
}

int inkling_index_load_shard(InklingIndex *index, const char *shard, const char *path)
{
    if (index == NULL || shard == NULL) {
        return 0;
    }
    InklingTensor *tensors = NULL;
    size_t count = 0;
    int success = inkling_safetensors_load_shard(path, &tensors, &count);
    if (success) {
        success = bind_tensors(index, shard, tensors, count);
    }
    free(tensors);
    return success;
}

void inkling_index_free(InklingIndex *index)
{
    if (index != NULL) {
        free(index->entries);
        free(index->slots);
        *index = (InklingIndex){0};
    }
}
