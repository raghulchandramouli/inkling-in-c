#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "inkling_model_internal.h"

static int failures;

static void expect(int condition, const char *message)
{
    if (!condition) { fprintf(stderr, "FAIL: %s\n", message); failures++; }
}

static int write_bytes(const char *path, const unsigned char *bytes, size_t count)
{
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(bytes, 1, count, file) == count;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static char *path_join(const char *directory, const char *name)
{
    size_t length = strlen(directory) + strlen(name) + 2;
    char *path = malloc(length);
    if (path) snprintf(path, length, "%s/%s", directory, name);
    return path;
}

static int copy_file(const char *source, const char *destination)
{
    FILE *input = fopen(source, "rb"), *output = fopen(destination, "wb");
    int ok = input && output;
    unsigned char buffer[4096];
    if (ok) {
        size_t count;
        while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0)
            if (fwrite(buffer, 1, count, output) != count) { ok = 0; break; }
        if (ferror(input)) ok = 0;
    }
    if (input && fclose(input) != 0) ok = 0;
    if (output && fclose(output) != 0) ok = 0;
    return ok;
}

static int copy_named(const char *source, const char *destination, const char *name)
{
    char *input = path_join(source, name), *output = path_join(destination, name);
    int ok = input && output && copy_file(input, output);
    free(input); free(output);
    return ok;
}

static void remove_fixture(const char *directory)
{
    DIR *stream = opendir(directory);
    if (!stream) return;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char *path = path_join(directory, entry->d_name);
        if (path) unlink(path);
        free(path);
    }
    closedir(stream);
    rmdir(directory);
}

static int copy_fixture(const char *directory, const InklingModel *model, char *temporary)
{
    if (!mkdtemp(temporary)) return 0;
    int ok = copy_named(directory, temporary, "config.json") &&
             copy_named(directory, temporary, "model.safetensors.index.json") &&
             copy_named(directory, temporary, "hf_quant_config.json");
    for (size_t i = 0; ok && i < model->shard_count; i++) {
        if (!model->shards[i].path) continue;
        const char *name = strrchr(model->shards[i].path, '/');
        ok = name && copy_named(directory, temporary, name + 1);
    }
    return ok;
}

static int patch_bytes(const char *path, uint64_t offset, const unsigned char *bytes, size_t count)
{
    if (offset > INT64_MAX) return 0;
    FILE *file = fopen(path, "r+b");
    if (!file) return 0;
    int ok = fseeko(file, (off_t)offset, SEEK_SET) == 0 && fwrite(bytes, 1, count, file) == count;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int patch_boolean(const char *path, const char *key, const char *replacement)
{
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    char bytes[65536];
    size_t count = fread(bytes, 1, sizeof(bytes) - 1, file);
    int ok = !ferror(file) && feof(file);
    fclose(file);
    bytes[count] = '\0';
    char *field = strstr(bytes, key), *start = field ? strchr(field, ':') : NULL;
    if (!ok || !start) return 0;
    start++;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') start++;
    char *end = start;
    while (*end && *end != ',' && *end != '}') end++;
    file = fopen(path, "wb");
    if (!file) return 0;
    size_t prefix = (size_t)(start - bytes);
    ok = fwrite(bytes, 1, prefix, file) == prefix && fputs(replacement, file) >= 0 && fputs(end, file) >= 0;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int replace_character(const char *path, const char *needle, unsigned char replacement)
{
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    char bytes[32768];
    size_t count = fread(bytes, 1, sizeof(bytes) - 1, file);
    int ok = !ferror(file) && feof(file);
    fclose(file);
    bytes[count] = '\0';
    char *name = strstr(bytes, needle);
    return ok && name && patch_bytes(path, (uint64_t)(name - bytes), &replacement, 1);
}

static int transpose_embedding(const char *path)
{
    char *header = NULL;
    uint64_t header_size;
    if (!inkling_safetensors_read_header(path, &header, &header_size)) return 0;
    char *name = strstr(header, "model.llm.embed.weight");
    char *shape = name ? strstr(name, "\"shape\"") : NULL;
    char *start = shape ? strchr(shape, '[') : NULL;
    char *end = start ? strchr(start, ']') : NULL;
    unsigned long long rows, columns;
    char replacement[80];
    int length = start && sscanf(start, "[%llu,%llu]", &rows, &columns) == 2 ?
        snprintf(replacement, sizeof(replacement), "%llu,%llu", columns, rows) : -1;
    int ok = end && length >= 0 && (size_t)length == (size_t)(end - start - 1) &&
        patch_bytes(path, 8 + (uint64_t)(start + 1 - header), (const unsigned char *)replacement, (size_t)length);
    free(header);
    return ok;
}

static int change_original_dtype(const char *path, const char *tensor_name)
{
    char *header = NULL;
    uint64_t header_size;
    if (!inkling_safetensors_read_header(path, &header, &header_size)) return 0;
    char *name = strstr(header, tensor_name);
    char *dtype = name ? strstr(name, "I64") : NULL;
    char *shape = name ? strstr(name, "\"shape\"") : NULL;
    char *count = shape ? strchr(shape, '3') : NULL;
    const unsigned char six = '6';
    int ok = dtype && count && patch_bytes(path, 8 + (uint64_t)(dtype - header),
        (const unsigned char *)"F32", 3) && patch_bytes(path, 8 + (uint64_t)(count - header), &six, 1);
    free(header);
    return ok;
}

static void malformed_fixture_checks(const char *directory, const InklingModel *source)
{
    const InklingTensor *original = NULL, *scale2 = NULL, *amax = NULL;
    for (uint64_t i = 0; i < source->index.count; i++) {
        const InklingTensor *tensor = &source->index.entries[i];
        if (strstr(tensor->name, ".original_shape")) original = tensor;
        if (strstr(tensor->name, ".scale2")) scale2 = tensor;
        if (strstr(tensor->name, ".input_amax")) amax = tensor;
    }
    static const char *labels[] = {"unsupported semantic switch rejects complete bind",
        "missing execution field rejects complete bind", "truncated shard rejects complete bind",
        "wrong original_shape payload rejects complete bind", "zero NVFP4 scale2 rejects complete bind",
        "zero NVFP4 input_amax rejects complete bind", "missing tensor rejects complete bind",
        "wrong matrix shape rejects complete bind", "wrong auxiliary dtype rejects complete bind", "missing quantization metadata rejects complete bind",
        "wrong quantization algorithm rejects complete bind", "wrong quantization group rejects complete bind",
        "quantization exclusion disagreement rejects complete bind"};
    for (size_t mutation = 0; mutation < sizeof(labels) / sizeof(labels[0]); mutation++) {
        if (((mutation >= 3 && mutation <= 5) || mutation == 8) && (!original || !scale2 || !amax)) continue;
        char temporary[] = "/tmp/inkling-bind-fixture-XXXXXX";
        int copied = copy_fixture(directory, source, temporary);
        expect(copied, "copy complete fixture for isolated rejection check");
        if (!copied) { remove_fixture(temporary); continue; }
        int patched = 0;
        char *path = NULL;
        if (mutation < 2) {
            path = path_join(temporary, "config.json");
            patched = path && patch_boolean(path, mutation ? "use_sconv" : "q_bias", mutation ? "null" : "true");
        } else if (mutation == 2) {
            const InklingShard *shard = &source->shards[0];
            const char *name = strrchr(shard->path, '/');
            path = name ? path_join(temporary, name + 1) : NULL;
            patched = path && truncate(path, (off_t)(shard->file_size - 1)) == 0;
        } else if (mutation == 6) {
            path = path_join(temporary, "model.safetensors.index.json");
            patched = path && replace_character(path, "model.llm.embed.weight", 'n');
        } else if (mutation == 7) {
            const InklingTensor *embedding = inkling_index_find_tensor(&source->index, "model.llm.embed.weight");
            path = embedding ? path_join(temporary, embedding->shard) : NULL;
            patched = path && transpose_embedding(path);
        } else if (mutation == 8) {
            path = path_join(temporary, original->shard);
            patched = path && change_original_dtype(path, original->name);
        } else if (mutation >= 9) {
            path = path_join(temporary, "hf_quant_config.json");
            if (mutation == 9) patched = path && unlink(path) == 0;
            if (mutation == 10) patched = path && patch_boolean(path, "quant_algo", "\"NVFP8\"");
            if (mutation == 11) patched = path && patch_boolean(path, "group_size", "32");
            if (mutation == 12) patched = path && replace_character(path, "model.llm.embed\"", 'n');
        } else {
            const InklingTensor *tensor = mutation == 3 ? original : mutation == 4 ? scale2 : amax;
            const InklingShard *shard = &source->shards[tensor->shard_id];
            path = path_join(temporary, tensor->shard);
            unsigned char zeros[8] = {0};
            size_t count = mutation == 3 ? 8 : mutation == 4 ? 4 : 2;
            patched = path && patch_bytes(path, 8 + shard->header_size + tensor->data_offset, zeros, count);
        }
        free(path);
        expect(patched, "malformed fixture mutation applied");
        if (patched) {
            InklingModel *model = (InklingModel *)(uintptr_t)1;
            expect(!inkling_model_open_text_fixture(temporary, &model) && model == NULL, labels[mutation]);
        }
        remove_fixture(temporary);
    }
}

static void reader_checks(void)
{
    char first[] = "/tmp/inkling-bind-first-XXXXXX", second[] = "/tmp/inkling-bind-second-XXXXXX";
    int first_fd = mkstemp(first), second_fd = mkstemp(second);
    expect(first_fd >= 0 && second_fd >= 0, "temporary range files exist");
    if (first_fd < 0 || second_fd < 0) {
        if (first_fd >= 0) { close(first_fd); unlink(first); }
        if (second_fd >= 0) { close(second_fd); unlink(second); }
        return;
    }
    close(first_fd); close(second_fd);
    const unsigned char original[] = {11, 23, 37, 41, 59, 67};
    expect(write_bytes(first, original, 3) && write_bytes(second, original + 3, 3), "range bytes written");
    InklingShard shard = {NULL, 8, 116};
    InklingBlob blobs[] = {{{0, 20, 3}, first}, {{0, 23, 3}, second}};
    InklingModel model = {0};
    model.shards = &shard; model.shard_count = 1;
    model.blobs = blobs; model.blob_count = 2; model.extracted = 1;
    InklingRange range = {0, 20, 6};
    unsigned char actual[6] = {0};
    expect(inkling_model_read(&model, range, 0, 6, actual) && memcmp(actual, original, 6) == 0,
           "adjacent chunks concatenate exactly");
    expect(inkling_model_read(&model, range, 1, 4, actual) && memcmp(actual, original + 1, 4) == 0,
           "read can start and end inside distinct chunks");
    expect(!inkling_model_read(&model, range, 1, 6, actual), "relative bounds reject overflow past tensor");
    expect(!inkling_model_read(&model, range, UINT64_MAX, 1, actual), "relative integer overflow rejected");
    InklingRange invalid = {0, UINT64_MAX - 1, 4};
    expect(!inkling_model_read(&model, invalid, 0, 1, actual), "absolute integer overflow rejected");
    invalid = range; invalid.shard = 1;
    expect(!inkling_model_read(&model, invalid, 0, 1, actual), "unknown shard rejected");
    invalid.start = 99; invalid.bytes = 2; invalid.shard = 0;
    expect(!inkling_model_read(&model, invalid, 0, 1, actual), "original shard bounds enforced");
    blobs[1].range.start++;
    expect(!inkling_model_read(&model, range, 0, 6, actual), "missing chunk byte fails explicitly");
    blobs[1].range.start--;
    expect(write_bytes(second, original + 3, 1), "truncate existing range file");
    expect(!inkling_model_read(&model, range, 0, 6, actual), "post-open truncation is a recoverable short read");
    expect(write_bytes(second, original + 3, 3), "restore range file");
    expect(inkling_model_read(&model, range, 0, 6, actual) && memcmp(actual, original, 6) == 0,
           "reader succeeds on retry after short read");
    model.extracted = 0;
    shard.path = first; shard.header_size = 0; shard.file_size = 14;
    const unsigned char ordinary[] = {0, 0, 0, 0, 0, 0, 0, 0, 11, 23, 37, 41, 59, 67};
    expect(write_bytes(first, ordinary, sizeof(ordinary)), "ordinary shard fixture written");
    range.start = 0;
    expect(inkling_model_read(&model, range, 0, 6, actual) && memcmp(actual, original, 6) == 0,
           "ordinary reader accounts for original header prefix");
    unlink(first); unlink(second);
}

int main(int argc, char **argv)
{
    reader_checks();
    InklingModel *model = (InklingModel *)(uintptr_t)1;
    expect(!inkling_model_open_text_fixture("/nonexistent/inkling-model", &model) && model == NULL,
           "failed constructor clears output");
    expect(!inkling_model_open_text_fixture(NULL, &model) && model == NULL, "NULL directory rejected");
    expect(!inkling_model_open_text_fixture(".", NULL), "NULL constructor output rejected");
    expect(inkling_model_logits_width(NULL) == 0, "NULL model has no logits width");
    inkling_model_close(NULL);
    if (argc >= 2) {
        int opened = inkling_model_open_text_fixture(argv[1], &model);
        expect(opened, "complete miniature model binds");
        if (opened) {
            expect(model->complete && model->layer_count == model->config.num_hidden_layers,
                   "complete binder publishes every configured layer");
            expect(inkling_model_logits_width(model) == model->config.unpadded_vocab_size,
                   "output vocabulary excludes padding rows");
            expect(model->embed_norm && model->final_norm, "complete model owns norm vectors");
            for (size_t i = 0; i < model->layer_count; i++) {
                const InklingLayer *layer = &model->layers[i];
                expect(layer->original_id == i && layer->q_norm && layer->k_norm && layer->relative &&
                       layer->k_conv && layer->v_conv && layer->attn_norm && layer->mlp_norm &&
                       layer->attn_conv && layer->mlp_conv, "layer binds required vectors under original ID");
            }
            malformed_fixture_checks(argv[1], model);
            inkling_model_close(model);
        }
        model = (InklingModel *)(uintptr_t)1;
        expect(!inkling_model_open(argv[1], &model) && model == NULL,
               "production opener retains pinned checkpoint identity gate");
    }
    if (failures) { fprintf(stderr, "%d model binding checks failed\n", failures); return 1; }
    puts("All model binding tests passed");
    return 0;
}
