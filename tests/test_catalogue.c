#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inkling/inkling.h"
#include "../src/io/inkling_json.h"

static int failures;

static void expect(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static int load_index(const char *json, InklingIndex *index)
{
    char path[] = "/tmp/inkling-catalogue-XXXXXX";
    int descriptor = mkstemp(path);
    if (descriptor < 0) {
        return 0;
    }
    size_t size = strlen(json);
    int success = write(descriptor, json, size) == (ssize_t)size;
    close(descriptor);
    if (success) {
        success = inkling_index_load(path, index);
    }
    unlink(path);
    return success;
}

static int parse_header(const char *json, uint64_t payload_size, InklingTensor **tensors, size_t *count)
{
    uint64_t length = (uint64_t)strlen(json);
    return inkling_safetensors_parse_header(json, length, 8 + length + payload_size, tensors, count);
}

static const struct {
    const char *name;
    InklingDataType dtype;
    uint64_t width;
    size_t pinned_count;
} types[] = {
    {"F32", INKLING_DTYPE_F32, 4, 158},
    {"BF16", INKLING_DTYPE_BF16, 2, 968},
    {"I64", INKLING_DTYPE_I64, 8, 78},
    {"F8_E4M3", INKLING_DTYPE_F8_E4M3, 1, 78},
    {"U8", INKLING_DTYPE_U8, 1, 78}
};

static void test_dtypes(void)
{
    for (size_t type = 0; type < sizeof(types) / sizeof(types[0]); type++) {
        char json[256];
        snprintf(json, sizeof(json),
                 "{\"w\":{\"dtype\":\"%s\",\"shape\":[2],\"data_offsets\":[0,%llu]}}",
                 types[type].name, (unsigned long long)(2 * types[type].width));
        InklingTensor *tensors = NULL;
        size_t count = 0;
        int success = parse_header(json, 2 * types[type].width, &tensors, &count);
        expect(success && count == 1, "stored dtype parses");
        if (success && count == 1) {
            expect(tensors[0].dtype == types[type].dtype && tensors[0].rank == 1 &&
                   tensors[0].shape[0] == 2 && tensors[0].byte_length == 2 * types[type].width,
                   "stored dtype width matches shape");
        }
        free(tensors);
    }
}

static void test_invalid_headers(void)
{
    static const char *cases[] = {
        "[]", "{} trailing", "{\"w\":{}}",
        "{\"w\":{\"dtype\":\"F4\",\"shape\":[2],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8x\",\"shape\":[1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\\u0000\",\"shape\":[1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":7,\"shape\":[1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":1,\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1.5],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[\"1\"],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[18446744073709551616],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[18446744073709551615,2],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"I64\",\"shape\":[2305843009213693952],\"data_offsets\":[0,0]}}",
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,3]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1,2]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[-1,0]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[64,65]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[18446744073709551614,18446744073709551615]}}",
        "{\"w\":{\"dtype\":\"U8\",\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1],\"extra\":0}}",
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"w\":{}}",
        "{\"w\\u0000x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
        "{\"__metadata__\":{\"format\":7}}",
        ("{\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[1,3]},"
         "\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}")
    };
    for (size_t test = 0; test < sizeof(cases) / sizeof(cases[0]); test++) {
        InklingTensor *tensors = NULL;
        size_t count = 99;
        expect(!parse_header(cases[test], 64, &tensors, &count), cases[test]);
        expect(tensors == NULL && count == 0, "invalid header clears outputs");
        free(tensors);
    }
    const char *valid = "{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    InklingTensor *tensors = NULL;
    size_t count = 0;
    expect(!inkling_safetensors_parse_header(valid, UINT64_MAX, UINT64_MAX, &tensors, &count),
           "header-size overflow rejected");
    expect(!inkling_safetensors_parse_header(valid, strlen(valid), 7, &tensors, &count),
           "file shorter than prefix rejected");
    expect(!inkling_safetensors_parse_header(valid, strlen(valid), strlen(valid) + 7, &tensors, &count),
           "file shorter than header rejected");
    expect(!parse_header(valid, 0, &tensors, &count), "truncated payload rejected");

    const char empty_and_scalar[] =
        "{\"b\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[0,2]},"
        "\"empty\":{\"dtype\":\"U8\",\"shape\":[0,3],\"data_offsets\":[1,1]},"
        "\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,3]},"
        "\"__metadata__\":{\"format\":\"pt\"}}";
    expect(parse_header(empty_and_scalar, 3, &tensors, &count) && count == 3,
           "scalar, zero-element tensor, touching ranges and metadata accepted");
    free(tensors);
}

static void test_index(void)
{
    InklingIndex index = {0};
    expect(load_index("{\"metadata\":{\"total_size\":3},\"weight_map\":{"
                      "\"a\":\"s\",\"i\":\"s\",\"q\":\"s\"}}", &index),
           "collision index loads");
    if (index.count != 3) {
        inkling_index_free(&index);
        return;
    }
    /* These FNV-1a hashes share their low 3 bits in the 8-slot table. */
    expect(index.slot_count == 8, "collision table has eight slots");
    for (size_t entry = 0; entry < 3; entry++) {
        expect(inkling_index_find_tensor(&index, index.entries[entry].name) == &index.entries[entry],
               "collision probing finds each entry");
    }
    expect(inkling_index_find_tensor(&index, "y") == NULL, "collision-chain miss terminates");
    expect(inkling_index_find_tensor(NULL, "a") == NULL &&
           inkling_index_find_tensor(&index, NULL) == NULL, "NULL lookup rejected");
    char shard[2];
    expect(!inkling_index_find_shard(&index, "a", shard, 1), "short shard buffer rejected");
    const char *header =
        "{\"q\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,3]},"
        "\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
        "\"i\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
    expect(inkling_index_bind_header(&index, "s", header, strlen(header), strlen(header) + 11),
           "complete shard binds");
    InklingTensor before = index.entries[0];
    expect(before.dtype == INKLING_DTYPE_U8 && before.byte_length == 1 && before.shard_id == 0,
           "bound catalogue carries metadata and shard ID");
    expect(!inkling_index_bind_header(&index, "s", header, strlen(header), strlen(header) + 10),
           "short shard fails binding");
    expect(memcmp(&before, &index.entries[0], sizeof(before)) == 0, "failed bind is atomic");
    expect(!inkling_index_bind_header(&index, "other", header, strlen(header), strlen(header) + 11),
           "wrong shard fails binding");
    const char *missing = "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    expect(!inkling_index_bind_header(&index, "s", missing, strlen(missing), strlen(missing) + 9),
           "missing indexed tensors rejected");
    char wrong_name[512];
    snprintf(wrong_name, sizeof(wrong_name), "%s", header);
    char *name = strstr(wrong_name, "\"i\"");
    if (name != NULL) {
        name[1] = 'x';
    }
    expect(!inkling_index_bind_header(&index, "s", wrong_name, strlen(wrong_name), strlen(wrong_name) + 11),
           "unknown header tensor rejected before publishing");
    expect(memcmp(&before, &index.entries[0], sizeof(before)) == 0, "name mismatch preserves catalogue");
    inkling_index_free(&index);
    expect(index.entries == NULL && index.slots == NULL && index.count == 0, "catalogue free clears state");
    inkling_index_free(&index);

    static const char *bad[] = {
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\":\"s\",\"a\":\"s\"}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\":\"s\",\"\\u0061\":\"s\"}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\":1}}",
        "{\"metadata\":{\"total_size\":-1},\"weight_map\":{\"a\":\"s\"}}",
        "{\"metadata\":{\"total_size\":18446744073709551616},\"weight_map\":{\"a\":\"s\"}}",
        "{\"other\":{\"total_size\":1},\"weight_map\":{\"a\":\"s\"}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\":\"../s\"}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\":\"s\\u0000x\"}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\\u0000x\":\"s\"}}",
        "{\"metadata\":{\"total_size\":1},\"weight_map\":{\"a\":\"s\"}} trailing"
    };
    for (size_t test = 0; test < sizeof(bad) / sizeof(bad[0]); test++) {
        expect(!load_index(bad[test], &index), "malformed index rejected");
        inkling_index_free(&index);
    }
}

static void test_actual_files(void)
{
    char path[] = "/tmp/inkling-shard-XXXXXX";
    int descriptor = mkstemp(path);
    expect(descriptor >= 0, "temporary shard created");
    if (descriptor < 0) {
        return;
    }
    const char *json = "{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}";
    uint64_t size = (uint64_t)strlen(json);
    unsigned char prefix[8], payload[2] = {17, 42};
    for (unsigned int byte = 0; byte < 8; byte++) {
        prefix[byte] = (unsigned char)(size >> (byte * 8));
    }
    expect(write(descriptor, prefix, sizeof(prefix)) == (ssize_t)sizeof(prefix) &&
           write(descriptor, json, (size_t)size) == (ssize_t)size &&
           write(descriptor, payload, sizeof(payload)) == (ssize_t)sizeof(payload), "complete shard written");
    close(descriptor);
    InklingTensor *tensors = NULL;
    size_t count = 0;
    expect(inkling_safetensors_load_shard(path, &tensors, &count) && count == 1,
           "actual file size accepts complete shard");
    free(tensors);
    InklingIndex index = {0};
    expect(load_index("{\"metadata\":{\"total_size\":2},\"weight_map\":{\"a\":\"s\"}}", &index),
           "actual-shard index loads");
    expect(inkling_index_load_shard(&index, "s", path), "actual shard binds into catalogue");
    inkling_index_free(&index);
    InklingTensorInfo info = {0};
    expect(inkling_safetensors_find_tensor(json, "a", &info), "raw payload metadata parses");
    unsigned char output[2] = {0};
    expect(inkling_safetensors_read_tensor_data(path, size, &info, output, sizeof(output)) &&
           memcmp(payload, output, sizeof(output)) == 0, "payload offset uses prefix plus header");
    expect(!inkling_safetensors_read_tensor_data(path, size + 1, &info, output, sizeof(output)),
           "incorrect supplied header length rejected");
    expect(!inkling_safetensors_read_tensor_data(path, UINT64_MAX, &info, output, sizeof(output)),
           "payload offset overflow rejected");
    expect(!inkling_safetensors_read_tensor_data(path, size, &info, output, 1), "short destination rejected");
    expect(truncate(path, (off_t)(8 + size + 1)) == 0, "shard payload truncated");
    expect(!inkling_safetensors_load_shard(path, &tensors, &count) && tensors == NULL && count == 0,
           "actual file size rejects truncated payload");
    expect(!inkling_safetensors_read_tensor_data(path, size, &info, output, sizeof(output)),
           "truncated payload read rejected");
    expect(truncate(path, (off_t)(8 + size - 1)) == 0, "shard header truncated");
    expect(!inkling_safetensors_load_shard(path, &tensors, &count), "truncated header rejected");
    expect(truncate(path, 7) == 0, "shard prefix truncated");
    expect(!inkling_safetensors_load_shard(path, &tensors, &count), "truncated prefix rejected");
    unlink(path);
}

static void test_pinned(const char *directory)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", directory);
    InklingIndex index = {0};
    expect(inkling_index_load(path, &index), "pinned catalogue loads");
    if (index.count == 0) {
        return;
    }
    expect(index.count == 1360 && index.num_shards == 10, "pinned index identity");
    size_t dtype_counts[5] = {0};
    uint64_t total_bytes = 0;
    uint64_t mtp_bytes = 0;
    for (uint32_t shard_id = 0; shard_id < index.num_shards; shard_id++) {
        const char *shard = NULL;
        for (uint64_t entry = 0; entry < index.count; entry++) {
            if (index.entries[entry].shard_id == shard_id) {
                shard = index.entries[entry].shard;
                break;
            }
        }
        expect(shard != NULL, "shard ID resolves");
        if (shard == NULL) {
            continue;
        }
        snprintf(path, sizeof(path), "%s/headers/%s", directory, shard);
        char *json = NULL;
        uint64_t header_size = 0;
        if (!inkling_safetensors_read_header(path, &json, &header_size)) {
            expect(0, "pinned header reads");
            continue;
        }
        InklingJsonDocument document = {0};
        expect(inkling_json_parse(json, (size_t)header_size, &document, NULL), "oracle header parses");
        uint64_t extent = 0;
        for (size_t member = 0; member < inkling_json_object_size(document.root); member++) {
            const InklingJsonValue *value = inkling_json_object_value_at(document.root, member);
            uint64_t end = 0;
            const InklingJsonValue *offsets = inkling_json_object_get(value, "data_offsets");
            if (inkling_json_number_u64(inkling_json_array_at(offsets, 1), &end) && end > extent) {
                extent = end;
            }
        }
        /* Fixtures contain no payload: use the declared extent explicitly.
         * Production load_shard must reject these same truncated files. */
        expect(inkling_index_bind_header(&index, shard, json, header_size, 8 + header_size + extent),
               "every pinned tensor validates and binds");
        expect(!inkling_index_load_shard(&index, shard, path), "header-only fixture is not a real shard");
        total_bytes += extent;
        if (strcmp(shard, "mtp.safetensors") == 0) {
            mtp_bytes = extent;
        }
        for (uint64_t entry = 0; entry < index.count; entry++) {
            const InklingTensor *tensor = &index.entries[entry];
            if (tensor->shard_id != shard_id) {
                continue;
            }
            expect(inkling_index_find_tensor(&index, tensor->name) == tensor, "all pinned hash lookups resolve");
            const InklingJsonValue *expected = inkling_json_object_get(document.root, tensor->name);
            const InklingJsonValue *shape = inkling_json_object_get(expected, "shape");
            const InklingJsonValue *offsets = inkling_json_object_get(expected, "data_offsets");
            uint64_t start = 0, end = 0;
            expect(inkling_json_number_u64(inkling_json_array_at(offsets, 0), &start) &&
                   inkling_json_number_u64(inkling_json_array_at(offsets, 1), &end) &&
                   tensor->data_offset == start && tensor->byte_length == end - start,
                   "catalogue offsets match pinned JSON");
            expect(tensor->rank == inkling_json_array_size(shape), "catalogue rank matches pinned JSON");
            for (uint32_t dimension = 0; dimension < tensor->rank; dimension++) {
                uint64_t size = 0;
                expect(inkling_json_number_u64(inkling_json_array_at(shape, dimension), &size) &&
                       tensor->shape[dimension] == size, "catalogue shape matches pinned JSON");
            }
            const char *dtype = inkling_json_string(inkling_json_object_get(expected, "dtype"), NULL);
            for (size_t type = 0; type < 5; type++) {
                if (dtype != NULL && strcmp(dtype, types[type].name) == 0) {
                    expect(tensor->dtype == types[type].dtype, "catalogue dtype matches pinned JSON");
                    dtype_counts[type]++;
                }
            }
        }
        inkling_json_document_free(&document);
        free(json);
    }
    for (size_t type = 0; type < 5; type++) {
        expect(dtype_counts[type] == types[type].pinned_count, "all pinned dtype counts match");
    }
    expect(total_bytes == index.total_size && mtp_bytes == UINT64_C(4463824912) &&
           total_bytes - mtp_bytes == UINT64_C(166269249680),
           "all-shard payload census matches index total, including MTP");
    const InklingTensor *weight = inkling_index_find_tensor(&index, "model.llm.layers.10.mlp.experts.w13_weight");
    const InklingTensor *shape = inkling_index_find_tensor(&index, "model.llm.layers.10.mlp.experts.w13_weight.original_shape");
    expect(weight != NULL && shape != NULL && weight->shard_id != shape->shard_id,
           "quantized auxiliaries resolve across shards");
    inkling_index_free(&index);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        return EXIT_FAILURE;
    }
    test_dtypes();
    test_invalid_headers();
    test_index();
    test_actual_files();
    test_pinned(argv[1]);
    if (failures != 0) {
        fprintf(stderr, "%d catalogue test(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    puts("All catalogue tests passed (1360 pinned tensors, 5 stored dtypes)");
    return EXIT_SUCCESS;
}
