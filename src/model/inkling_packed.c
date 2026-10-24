#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "inkling_model_internal.h"
#include "inkling_alloc.h"
#include "inkling_json.h"

#define PACKED_HEADER_LIMIT (100U * 1024U * 1024U)
#define SOURCE_REVISION "b6a99534467840620d411e4cd4ad5819b2610d9c"

typedef struct { uint32_t state[8]; uint64_t bytes; unsigned char block[64]; size_t used; } Sha256;
typedef struct { InklingRange source; uint64_t destination; } PackedRange;
struct InklingPackedTrunk { int fd; PackedRange *ranges; size_t count; };

static uint32_t rotate(uint32_t x, unsigned int n) { return (x >> n) | (x << (32U - n)); }

static void sha_block(Sha256 *hash, const unsigned char *block)
{
    static const uint32_t constants[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t words[64];
    for (size_t i = 0; i < 16; i++) {
        const unsigned char *p = block + i * 4;
        words[i] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    }
    for (size_t i = 16; i < 64; i++) {
        uint32_t a = words[i - 15], b = words[i - 2];
        words[i] = words[i - 16] + (rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3)) + words[i - 7] +
                   (rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10));
    }
    uint32_t a=hash->state[0],b=hash->state[1],c=hash->state[2],d=hash->state[3];
    uint32_t e=hash->state[4],f=hash->state[5],g=hash->state[6],h=hash->state[7];
    for (size_t i = 0; i < 64; i++) {
        uint32_t first = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + constants[i] + words[i];
        uint32_t second = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
    }
    hash->state[0]+=a; hash->state[1]+=b; hash->state[2]+=c; hash->state[3]+=d;
    hash->state[4]+=e; hash->state[5]+=f; hash->state[6]+=g; hash->state[7]+=h;
}

static Sha256 sha_start(void)
{
    Sha256 hash = {{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19},0,{0},0};
    return hash;
}

static void sha_update(Sha256 *hash, const void *data, size_t bytes)
{
    const unsigned char *input = data;
    hash->bytes += bytes;
    while (bytes) {
        size_t take = 64 - hash->used;
        if (take > bytes) take = bytes;
        memcpy(hash->block + hash->used, input, take);
        hash->used += take; input += take; bytes -= take;
        if (hash->used == 64) { sha_block(hash, hash->block); hash->used = 0; }
    }
}

static void sha_finish(Sha256 *hash, char hex[65])
{
    uint64_t bits = hash->bytes * 8;
    unsigned char suffix[72] = {0x80};
    size_t padding = hash->used < 56 ? 56 - hash->used : 120 - hash->used;
    for (size_t i = 0; i < 8; i++) suffix[padding + i] = (unsigned char)(bits >> (56 - i * 8));
    sha_update(hash, suffix, padding + 8);
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        unsigned char byte = (unsigned char)(hash->state[i / 4] >> (24 - (i % 4) * 8));
        hex[i * 2] = digits[byte >> 4]; hex[i * 2 + 1] = digits[byte & 15];
    }
    hex[64] = '\0';
}

static int packed_fail(void)
{
    fprintf(stderr, "model: invalid packed trunk (format, source identity, coverage or SHA256)\n");
    return 0;
}

static int positional_read(int fd, uint64_t offset, size_t bytes, void *destination)
{
    if (offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset) return 0;
    unsigned char *output = destination;
    while (bytes) {
        size_t take = bytes > 1024 * 1024 ? 1024 * 1024 : bytes;
        off_t position = (off_t)offset;
        if (position < 0 || (uint64_t)position != offset) return 0;
        ssize_t got = pread(fd, output, take, position);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        bytes -= (size_t)got; output += got; offset += (uint64_t)got;
    }
    return 1;
}

static int hash_range(Sha256 *hash, int fd, uint64_t start, uint64_t bytes)
{
    unsigned char buffer[65536];
    while (bytes) {
        size_t take = bytes > sizeof(buffer) ? sizeof(buffer) : (size_t)bytes;
        if (!positional_read(fd, start, take, buffer)) return 0;
        sha_update(hash, buffer, take);
        start += take; bytes -= take;
    }
    return 1;
}

static int source_identity(const char *directory, char hex[65])
{
    static const char *names[] = {"config.json", "model.safetensors.index.json", "hf_quant_config.json"};
    Sha256 hash = sha_start();
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        size_t total;
        if (!inkling_size_add(strlen(directory), strlen(names[i]) + 2, &total)) return 0;
        char *path = malloc(total);
        if (!path) return 0;
        snprintf(path, total, "%s/%s", directory, names[i]);
        int fd = open(path, O_RDONLY);
        free(path);
        struct stat status;
        int ok = fd >= 0 && fstat(fd, &status) == 0 && S_ISREG(status.st_mode) &&
                 status.st_size >= 0 && (uint64_t)status.st_size <= PACKED_HEADER_LIMIT;
        if (ok) {
            uint64_t bytes = (uint64_t)status.st_size;
            unsigned char length[8];
            for (size_t j = 0; j < 8; j++) length[j] = (unsigned char)(bytes >> (8 * j));
            sha_update(&hash, names[i], strlen(names[i]) + 1);
            sha_update(&hash, length, sizeof(length));
            ok = hash_range(&hash, fd, 0, bytes);
        }
        if (fd >= 0) close(fd);
        if (!ok) return 0;
    }
    sha_finish(&hash, hex);
    return 1;
}

static int string_equal(const InklingJsonValue *value, const char *expected)
{
    size_t length;
    const char *text = inkling_json_string(value, &length);
    return text && length == strlen(expected) && memcmp(text, expected, length) == 0;
}

static int header_identity(const char *header, size_t bytes, const InklingJsonValue *metadata)
{
    static const char field[] = "\"header_sha256\":\"";
    const char *value = strstr(header, field);
    if (!value || strstr(value + 1, field)) return 0;
    value += sizeof(field) - 1;
    size_t offset = (size_t)(value - header);
    if (offset > bytes || bytes - offset < 65 || value[64] != '\"') return 0;
    Sha256 hash = sha_start();
    static const char zeros[] = "0000000000000000000000000000000000000000000000000000000000000000";
    sha_update(&hash, header, offset);
    sha_update(&hash, zeros, 64);
    sha_update(&hash, value + 64, bytes - offset - 64);
    char digest[65];
    sha_finish(&hash, digest);
    return string_equal(inkling_json_object_get(metadata, "header_sha256"), digest);
}

static int trunk_name(const char *name)
{
    return strncmp(name, "model.llm.", 10) == 0 && strstr(name, ".mlp.experts.") == NULL;
}

static int compare_ranges(const void *left, const void *right)
{
    const PackedRange *a = left, *b = right;
    if (a->source.shard != b->source.shard) return a->source.shard < b->source.shard ? -1 : 1;
    return a->source.start < b->source.start ? -1 : a->source.start > b->source.start;
}

void inkling_packed_close(struct InklingPackedTrunk *packed)
{
    if (!packed) return;
    close(packed->fd);
    free(packed->ranges);
    free(packed);
}

int inkling_packed_open(InklingModel *model, const char *directory, const char *path)
{
    struct InklingPackedTrunk *packed = calloc(1, sizeof(*packed));
    if (!packed) return packed_fail();
    packed->fd = open(path, O_RDONLY);
    struct stat status;
    char *header = NULL;
    InklingTensor *tensors = NULL;
    InklingJsonDocument document = {0}, sources = {0};
    unsigned char prefix[8];
    size_t count = 0;
    int ok = packed->fd >= 0 && fstat(packed->fd, &status) == 0 && S_ISREG(status.st_mode) && status.st_size >= 8 &&
             positional_read(packed->fd, 0, sizeof(prefix), prefix);
    if (!ok) goto done;
    uint64_t header_size = 0, file_size = (uint64_t)status.st_size;
    for (size_t i = 0; i < 8; i++) header_size |= (uint64_t)prefix[i] << (8 * i);
    ok = header_size > 0 && header_size <= PACKED_HEADER_LIMIT && header_size <= file_size - 8 && header_size < SIZE_MAX;
    if (!ok) goto done;
    header = malloc((size_t)header_size + 1);
    ok = header && positional_read(packed->fd, 8, (size_t)header_size, header);
    if (!ok) goto done;
    header[header_size] = '\0';
    ok = inkling_safetensors_parse_header(header, header_size, file_size, &tensors, &count) && count &&
         inkling_json_parse(header, (size_t)header_size, &document, NULL);
    if (!ok) goto done;
    const InklingJsonValue *metadata = inkling_json_object_get(document.root, "__metadata__");
    char identity[65], payload[65];
    Sha256 hash = sha_start();
    ok = inkling_json_object_size(metadata) == 6 && header_identity(header, (size_t)header_size, metadata) &&
         string_equal(inkling_json_object_get(metadata, "schema"), "inkling-trunk-v1") &&
         string_equal(inkling_json_object_get(metadata, "checkpoint_revision"), SOURCE_REVISION) &&
         source_identity(directory, identity) && string_equal(inkling_json_object_get(metadata, "source_sha256"), identity) &&
         hash_range(&hash, packed->fd, 8 + header_size, file_size - 8 - header_size);
    if (!ok) goto done;
    sha_finish(&hash, payload);
    ok = string_equal(inkling_json_object_get(metadata, "payload_sha256"), payload);
    size_t mapping_length = 0;
    const char *mapping = inkling_json_string(inkling_json_object_get(metadata, "source_ranges"), &mapping_length);
    ok = ok && mapping && inkling_json_parse(mapping, mapping_length, &sources, NULL) &&
         inkling_json_type(sources.root) == INKLING_JSON_OBJECT && inkling_json_object_size(sources.root) == count;
    if (!ok) goto done;
    size_t expected = 0;
    for (uint64_t i = 0; i < model->index.count; i++) if (trunk_name(model->index.entries[i].name)) expected++;
    void *allocation = NULL;
    ok = expected == count && inkling_alloc_array(count, sizeof(*packed->ranges), &allocation);
    if (!ok) goto done;
    packed->ranges = allocation;
    packed->count = count;
    for (size_t i = 0; i < count; i++) {
        const InklingTensor *tensor = &tensors[i];
        const InklingTensor *original = inkling_index_find_tensor(&model->index, tensor->name);
        const InklingJsonValue *source = inkling_json_object_get(sources.root, tensor->name);
        uint64_t start, bytes;
        ok = original && trunk_name(tensor->name) && tensor->dtype == original->dtype && tensor->rank == original->rank &&
             tensor->byte_length == original->byte_length && inkling_json_type(source) == INKLING_JSON_ARRAY &&
             inkling_json_array_size(source) == 3 && string_equal(inkling_json_array_at(source, 0), original->shard) &&
             inkling_json_number_u64(inkling_json_array_at(source, 1), &start) && start == original->data_offset &&
             inkling_json_number_u64(inkling_json_array_at(source, 2), &bytes) && bytes == original->byte_length;
        if (!ok) goto done;
        for (uint32_t j = 0; j < tensor->rank; j++) if (tensor->shape[j] != original->shape[j]) { ok = 0; goto done; }
        packed->ranges[i].source = (InklingRange){original->shard_id, start, bytes};
        packed->ranges[i].destination = 8 + header_size + tensor->data_offset;
    }
    qsort(packed->ranges, count, sizeof(*packed->ranges), compare_ranges);
    for (size_t i = 1; i < count; i++) {
        const InklingRange a = packed->ranges[i - 1].source, b = packed->ranges[i].source;
        if (a.shard == b.shard && b.start - a.start < a.bytes) { ok = 0; goto done; }
    }
    model->packed = packed;
    packed = NULL;
done:
    free(header); free(tensors);
    inkling_json_document_free(&document); inkling_json_document_free(&sources);
    inkling_packed_close(packed);
    return ok || packed_fail();
}

int inkling_packed_read(const InklingModel *model, InklingRange range, uint64_t relative,
                         size_t bytes, void *destination, int *handled)
{
    *handled = 0;
    const struct InklingPackedTrunk *packed = model->packed;
    if (!packed) return 1;
    uint64_t start = range.start + relative;
    size_t low = 0, high = packed->count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        const InklingRange candidate = packed->ranges[middle].source;
        if (candidate.shard < range.shard || (candidate.shard == range.shard && candidate.start <= start)) low = middle + 1;
        else high = middle;
    }
    if (!low) return 1;
    const PackedRange *entry = &packed->ranges[low - 1];
    if (entry->source.shard != range.shard || start < entry->source.start || start - entry->source.start >= entry->source.bytes) return 1;
    *handled = 1;
    uint64_t within = start - entry->source.start;
    if (bytes > entry->source.bytes - within || !positional_read(packed->fd, entry->destination + within, bytes, destination)) {
        fprintf(stderr, "model: packed trunk read failed\n");
        return 0;
    }
    return 1;
}
