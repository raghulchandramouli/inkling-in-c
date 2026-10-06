#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "inkling/inkling.h"
#include "inkling_json.h"

#define MODEL "thinkingmachines/Inkling-Small-NVFP4"
#define REVISION "b6a99534467840620d411e4cd4ad5819b2610d9c"

/* The pinned header fixtures have no payload. These lengths are the census
 * of their original shards, not inferred from whichever header we are given. */
static const uint64_t payload_sizes[] = {
    UINT64_C(19797832444), UINT64_C(18832935046), UINT64_C(19460467600),
    UINT64_C(19886191036), UINT64_C(19970064838), UINT64_C(18910717126),
    UINT64_C(15485554034), UINT64_C(19791436922), UINT64_C(14134050634),
    UINT64_C(4463824912)
};

static const char *dtype_names[] = {"UNKNOWN", "F32", "BF16", "I64", "F8_E4M3", "U8"};
static const char *class_names[] = {
    "trunk", "routed-packed", "scales-and-auxiliaries", "embeddings-unembed",
    "vision", "audio", "mtp"
};

typedef struct {
    InklingIndex index;
    unsigned char *seen;
    int skip_mtp;
    int names_only;
    uint64_t counts[7], bytes[7];
} Census;

static int path_join(char *path, size_t size, const char *directory, const char *name)
{
    int length = snprintf(path, size, "%s/%s", directory, name);
    if (length < 0 || (size_t)length >= size) {
        fputs("verify: model path is too long\n", stderr);
        return 0;
    }
    return 1;
}

static int load_json(const char *path, InklingJsonDocument *document)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "verify: cannot open %s: %s\n", path, strerror(errno));
        return 0;
    }
    int ok = 0;
    char *text = NULL;
    if (fseek(file, 0, SEEK_END) != 0) goto done;
    long size = ftell(file);
    if (size < 0 || size > 16 * 1024 * 1024 || fseek(file, 0, SEEK_SET) != 0) goto done;
    text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) goto done;
    InklingJsonError error;
    ok = inkling_json_parse(text, (size_t)size, document, &error);
    if (!ok) fprintf(stderr, "verify: %s:%zu:%zu: %s\n", path, error.line, error.column, error.message);
done:
    if (!ok) fprintf(stderr, "verify: cannot load JSON metadata %s\n", path);
    free(text);
    fclose(file);
    return ok;
}

static const InklingJsonValue *get(const InklingJsonValue *value, const char *key)
{
    return inkling_json_object_get(value, key);
}

static int string_is(const InklingJsonValue *value, const char *expected)
{
    size_t size = 0;
    const char *text = inkling_json_string(value, &size);
    return text && size == strlen(expected) && memcmp(text, expected, size) == 0;
}

static int number_is(const InklingJsonValue *value, double expected)
{
    double number;
    return inkling_json_number_double(value, &number) && number == expected;
}

/* This executable supports one pinned architecture, not arbitrary Inkling
 * variants. Validate shape-affecting and execution-affecting config fields. */
static int config_contract(const InklingJsonValue *root)
{
    static const struct { const char *section, *key; double value; } numbers[] = {
        {NULL,"eos_token_id",200006},
        {"text_config","model_max_length",1048576}, {"text_config","hidden_size",4096},
        {"text_config","num_hidden_layers",42}, {"text_config","vocab_size",201024},
        {"text_config","num_attention_heads",32}, {"text_config","num_key_value_heads",8},
        {"text_config","head_dim",128}, {"text_config","d_rel",16}, {"text_config","rel_extent",1024},
        {"text_config","log_scaling_n_floor",128000}, {"text_config","log_scaling_alpha",0.1},
        {"text_config","rms_norm_eps",1e-6}, {"text_config","dense_mlp_idx",2},
        {"text_config","sconv_kernel_size",4}, {"text_config","unpadded_vocab_size",200058},
        {"text_config","logits_mup_width_multiplier",16}, {"text_config","swa_head_dim",128},
        {"text_config","swa_num_attention_heads",32}, {"text_config","swa_num_key_value_heads",8},
        {"text_config","sliding_window_size",512}, {"text_config","n_routed_experts",256},
        {"text_config","num_experts_per_tok",6}, {"text_config","n_shared_experts",2},
        {"text_config","dense_intermediate_size",16384}, {"text_config","intermediate_size",2048},
        {"text_config","route_scale",8},
        {"audio_config","decoder_dmodel",4096}, {"audio_config","n_mel_bins",80},
        {"audio_config","mel_vocab_size",16}, {"audio_config","dmel_min_value",-7},
        {"audio_config","dmel_max_value",2},
        {"vision_config","decoder_dmodel",4096}, {"vision_config","patch_size",40},
        {"vision_config","temporal_patch_size",2}, {"vision_config","n_channels",3},
        {"vision_config","n_layers",4}, {"mtp_config","num_nextn_predict_layers",8}
    };
    static const struct { const char *section, *key; int value; } flags[] = {
        {"text_config","q_bias",0}, {"text_config","o_bias",0},
        {"text_config","use_embed_norm",1}, {"text_config","use_sconv",1},
        {"text_config","shared_expert_sink",1}, {"text_config","use_gate_bias",1},
        {"text_config","norm_after_topk",1}, {"text_config","use_global_scale",1},
        {"audio_config","bias",0}, {"audio_config","use_audio_norm",1},
        {"vision_config","use_vision_norm",1}, {"mtp_config","chain_hidden_post_norm",0}
    };
    static const struct { const char *section, *key, *value; } strings[] = {
        {NULL,"model_type","inkling_mm_model"}, {"text_config","torch_dtype","bfloat16"},
        {"text_config","gate_activation","sigmoid"}, {"audio_config","audio_mode","dmel"},
        {"vision_config","vision_encoder_type","hmlp"}
    };
    for (size_t i = 0; i < sizeof(numbers) / sizeof(numbers[0]); i++) {
        const InklingJsonValue *section = numbers[i].section ? get(root, numbers[i].section) : root;
        if (!number_is(get(section, numbers[i].key), numbers[i].value)) {
            fprintf(stderr, "verify: unsupported config %s.%s (expected %.9g)\n",
                    numbers[i].section ? numbers[i].section : "root", numbers[i].key, numbers[i].value);
            return 0;
        }
    }
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); i++) {
        int flag;
        if (!inkling_json_boolean(get(get(root, flags[i].section), flags[i].key), &flag) || flag != flags[i].value) {
            fprintf(stderr, "verify: unsupported config %s.%s\n", flags[i].section, flags[i].key);
            return 0;
        }
    }
    for (size_t i = 0; i < sizeof(strings) / sizeof(strings[0]); i++) {
        const InklingJsonValue *section = strings[i].section ? get(root, strings[i].section) : root;
        if (!string_is(get(section, strings[i].key), strings[i].value)) {
            fprintf(stderr, "verify: unsupported config %s\n", strings[i].key);
            return 0;
        }
    }
    const InklingJsonValue *architectures = get(root, "architectures");
    const InklingJsonValue *softcap = get(get(root, "text_config"), "final_logit_softcapping");
    if (inkling_json_type(architectures) != INKLING_JSON_ARRAY || inkling_json_array_size(architectures) != 1 ||
        !string_is(inkling_json_array_at(architectures, 0), "InklingForConditionalGeneration") ||
        !softcap || inkling_json_type(softcap) != INKLING_JSON_NULL) {
        fputs("verify: unsupported architecture or final_logit_softcapping\n", stderr);
        return 0;
    }
    return 1;
}

static int local_layer(const InklingJsonValue *ids, uint32_t layer)
{
    for (size_t i = 0; i < inkling_json_array_size(ids); i++) {
        uint64_t id;
        if (inkling_json_number_u64(inkling_json_array_at(ids, i), &id) && id == layer) return 1;
    }
    return 0;
}

static int layer_contract(const InklingJsonValue *ids, int mtp)
{
    unsigned char seen[42] = {0};
    size_t limit = mtp ? 8 : 42, expected = mtp ? 6 : 35;
    if (inkling_json_type(ids) != INKLING_JSON_ARRAY || inkling_json_array_size(ids) != expected) return 0;
    for (size_t i = 0; i < expected; i++) {
        uint64_t id;
        if (!inkling_json_number_u64(inkling_json_array_at(ids, i), &id) || id >= limit || seen[id]) return 0;
        seen[id] = 1;
    }
    for (size_t i = 0; i < limit; i++) {
        int is_local = mtp ? (i != 1 && i != 3) : (i == 0 || i % 6 != 5);
        /* Global text layers are 5,11,17,23,29,35,41. */
        if (seen[i] != is_local) return 0;
    }
    return 1;
}

static int excluded(const InklingJsonValue *modules, const char *name)
{
    for (size_t i = 0; i < inkling_json_array_size(modules); i++) {
        size_t size;
        const char *prefix = inkling_json_string(inkling_json_array_at(modules, i), &size);
        if (prefix && strlen(name) >= size && memcmp(name, prefix, size) == 0 &&
            (name[size] == '\0' || name[size] == '.')) return 1;
    }
    return 0;
}

static int quant_contract(const InklingJsonValue *quant)
{
    const InklingJsonValue *modules = get(quant, "exclude_modules");
    if (!string_is(get(quant, "quant_algo"), "NVFP4") ||
        !string_is(get(quant, "kv_cache_quant_algo"), "none") ||
        !number_is(get(quant, "group_size"), 16) ||
        inkling_json_type(modules) != INKLING_JSON_ARRAY || inkling_json_array_size(modules) != 312) goto invalid;
    for (size_t i = 0; i < inkling_json_array_size(modules); i++) {
        size_t size;
        const char *prefix = inkling_json_string(inkling_json_array_at(modules, i), &size);
        if (!prefix || !size || strlen(prefix) != size) goto invalid;
        for (size_t j = 0; j < i; j++)
            if (string_is(inkling_json_array_at(modules, j), prefix)) goto invalid;
    }
    const InklingJsonValue *modelopt = get(quant, "modelopt_quant_config");
    const InklingJsonValue *cfg = get(modelopt, "quant_cfg");
    if (!string_is(get(modelopt, "algorithm"), "max")) goto invalid;
    const char *keys[] = {"*weight_quantizer", "*input_quantizer"};
    for (size_t i = 0; i < 2; i++) {
        const InklingJsonValue *entry = get(cfg, keys[i]);
        const InklingJsonValue *bits = get(entry, "num_bits"), *block = get(entry, "block_sizes");
        const InklingJsonValue *scale = get(block, "scale_bits"), *axis = get(entry, "axis");
        int enabled;
        if (inkling_json_array_size(bits) != 2 || !number_is(inkling_json_array_at(bits, 0), 2) ||
            !number_is(inkling_json_array_at(bits, 1), 1) || !number_is(get(block, "-1"), 16) ||
            !string_is(get(block, "type"), "dynamic") || inkling_json_array_size(scale) != 2 ||
            !number_is(inkling_json_array_at(scale, 0), 4) || !number_is(inkling_json_array_at(scale, 1), 3) ||
            !axis || inkling_json_type(axis) != INKLING_JSON_NULL ||
            !inkling_json_boolean(get(entry, "enable"), &enabled) || !enabled) goto invalid;
    }
    int enabled;
    if (!inkling_json_boolean(get(get(cfg, "default"), "enable"), &enabled) || enabled) goto invalid;
    return 1;
invalid:
    fputs("verify: unsupported hf_quant_config.json: expected pinned NVFP4/group_size=16 and exclusions\n", stderr);
    return 0;
}

static int require_tensor(Census *census, const char *prefix, const char *suffix,
                          InklingDataType dtype, uint32_t rank, uint64_t a, uint64_t b, uint64_t c, size_t class_id)
{
    char name[INKLING_INDEX_MAX_NAME_LENGTH];
    int length = snprintf(name, sizeof(name), "%s%s", prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(name)) return 0;
    const InklingTensor *tensor = inkling_index_find_tensor(&census->index, name);
    if (!tensor) {
        fprintf(stderr, "verify: missing required tensor %s (check global index and shard header)\n", name);
        return 0;
    }
    census->seen[(size_t)(tensor - census->index.entries)] = 1;
    if (class_id == 6 && strcmp(tensor->shard, "mtp.safetensors") != 0) {
        fprintf(stderr, "verify: MTP tensor %s must be in mtp.safetensors\n", name);
        return 0;
    }
    if (class_id != 6 && strcmp(tensor->shard, "mtp.safetensors") == 0) {
        fprintf(stderr, "verify: non-MTP tensor %s in optional MTP shard\n", name);
        return 0;
    }
    if (census->names_only || (census->skip_mtp && class_id == 6)) return 1;
    uint64_t shape[] = {a, b, c};
    int matches = tensor->dtype == dtype && tensor->rank == rank;
    for (uint32_t i = 0; matches && i < rank; i++) matches = tensor->shape[i] == shape[i];
    if (!matches) {
        fprintf(stderr, "verify: dtype/shape mismatch for %s: expected %s [", name, dtype_names[dtype]);
        for (uint32_t i = 0; i < rank; i++) fprintf(stderr, "%s%" PRIu64, i ? "," : "", shape[i]);
        fprintf(stderr, "]; got %s [", dtype_names[tensor->dtype]);
        for (uint32_t i = 0; i < tensor->rank; i++) fprintf(stderr, "%s%" PRIu64, i ? "," : "", tensor->shape[i]);
        fputs("]\n", stderr);
        return 0;
    }
    census->counts[class_id]++;
    census->bytes[class_id] += tensor->byte_length;
    return 1;
}

static int block_contract(Census *census, const char *prefix, int local, size_t class_id)
{
    static const struct { const char *suffix; uint32_t rank; uint64_t a, b, c; } families[] = {
        {"attn.k_norm.weight",1,128,0,0}, {"attn.q_norm.weight",1,128,0,0},
        {"attn.k_sconv.weight",3,1024,1,4}, {"attn.v_sconv.weight",3,1024,1,4},
        {"attn.wk_dv.weight",2,1024,4096,0}, {"attn.wv_dv.weight",2,1024,4096,0},
        {"attn.wo_ud.weight",2,4096,4096,0}, {"attn.wq_du.weight",2,4096,4096,0},
        {"attn.wr_du.weight",2,512,4096,0}, {"attn_norm.weight",1,4096,0,0},
        {"mlp_norm.weight",1,4096,0,0}, {"attn_sconv.weight",3,4096,1,4},
        {"mlp_sconv.weight",3,4096,1,4}
    };
    for (size_t i = 0; i < sizeof(families) / sizeof(families[0]); i++)
        if (!require_tensor(census, prefix, families[i].suffix, INKLING_DTYPE_BF16,
                            families[i].rank, families[i].a, families[i].b, families[i].c, class_id)) return 0;
    return require_tensor(census, prefix, "attn.rel_logits_proj.proj", INKLING_DTYPE_BF16,
                          2, 16, local ? 512 : 1024, 0, class_id);
}

static int dense_contract(Census *census, const char *prefix, size_t class_id)
{
    return require_tensor(census,prefix,"mlp.global_scale",INKLING_DTYPE_BF16,1,1,0,0,class_id) &&
           require_tensor(census,prefix,"mlp.w13_dn.weight",INKLING_DTYPE_BF16,2,32768,4096,0,class_id) &&
           require_tensor(census,prefix,"mlp.w2_md.weight",INKLING_DTYPE_BF16,2,4096,16384,0,class_id);
}

static int routed_contract(Census *census, const char *prefix, const InklingJsonValue *modules)
{
    const char *weights[] = {"mlp.experts.w13_weight", "mlp.experts.w2_weight"};
    for (size_t i = 0; i < 2; i++) {
        char name[INKLING_INDEX_MAX_NAME_LENGTH];
        snprintf(name, sizeof(name), "%s%s", prefix, weights[i]);
        int packed = !excluded(modules, name);
        /* Only routed layer 2 is excluded in the pinned checkpoint. */
        if (packed != (strcmp(prefix, "model.llm.layers.2.") != 0)) {
            fprintf(stderr, "verify: unsupported routed exclusion for %s\n", name);
            return 0;
        }
        uint64_t last = i ? 2048 : 4096;
        if (!require_tensor(census,prefix,weights[i],packed ? INKLING_DTYPE_U8 : INKLING_DTYPE_BF16,
                            3,256,4096,packed ? last / 2 : last,packed ? 1 : 0)) return 0;
        if (packed &&
            (!require_tensor(census,name,".input_amax",INKLING_DTYPE_BF16,1,1,0,0,2) ||
             !require_tensor(census,name,".original_shape",INKLING_DTYPE_I64,1,3,0,0,2) ||
             !require_tensor(census,name,".scale",INKLING_DTYPE_F8_E4M3,3,256,4096,last/16,2) ||
             !require_tensor(census,name,".scale2",INKLING_DTYPE_F32,1,256,0,0,2))) return 0;
    }
    return require_tensor(census,prefix,"mlp.gate.bias",INKLING_DTYPE_F32,1,256,0,0,0) &&
           require_tensor(census,prefix,"mlp.gate.global_scale",INKLING_DTYPE_F32,1,1,0,0,0) &&
           require_tensor(census,prefix,"mlp.gate.weight",INKLING_DTYPE_BF16,2,258,4096,0,0) &&
           require_tensor(census,prefix,"mlp.shared_experts.shared_w13_weight",INKLING_DTYPE_BF16,3,2,4096,4096,0) &&
           require_tensor(census,prefix,"mlp.shared_experts.shared_w2_weight",INKLING_DTYPE_BF16,3,2,4096,2048,0);
}

static int tensor_contract(Census *census, const InklingJsonValue *config, const InklingJsonValue *quant)
{
    const InklingJsonValue *text_ids = get(get(config,"text_config"),"local_layer_ids");
    const InklingJsonValue *mtp_ids = get(get(config,"mtp_config"),"local_layer_ids");
    if (!require_tensor(census,"model.llm.","embed.weight",INKLING_DTYPE_BF16,2,201024,4096,0,3) ||
        !require_tensor(census,"model.llm.","unembed.weight",INKLING_DTYPE_BF16,2,201024,4096,0,3) ||
        !require_tensor(census,"model.llm.","embed_norm.weight",INKLING_DTYPE_BF16,1,4096,0,0,0) ||
        !require_tensor(census,"model.llm.","norm.weight",INKLING_DTYPE_BF16,1,4096,0,0,0)) return 0;
    char prefix[80];
    for (uint32_t layer = 0; layer < 42; layer++) {
        snprintf(prefix,sizeof(prefix),"model.llm.layers.%" PRIu32 ".",layer);
        if (!block_contract(census,prefix,local_layer(text_ids,layer),0)) return 0;
        if (layer < 2) {
            if (!dense_contract(census,prefix,0)) return 0;
        } else if (!routed_contract(census,prefix,get(quant,"exclude_modules"))) return 0;
    }
    if (!require_tensor(census,"model.audio.","encoder.weight",INKLING_DTYPE_BF16,2,1280,4096,0,5) ||
        !require_tensor(census,"model.audio.","final_norm.weight",INKLING_DTYPE_BF16,1,4096,0,0,5) ||
        !require_tensor(census,"model.visual.","final_norm.weight",INKLING_DTYPE_BF16,1,4096,0,0,4)) return 0;
    /* HMLP stage shapes are the pinned patch-40/temporal-2/channel-3 contract. */
    const uint64_t rows[] = {128,320,4800,4096}, columns[] = {75,512,5120,9600};
    for (uint32_t stage = 0; stage < 4; stage++) {
        snprintf(prefix,sizeof(prefix),"model.visual.layers.linear_%" PRIu32,stage);
        if (!require_tensor(census,prefix,".weight",INKLING_DTYPE_BF16,2,rows[stage],columns[stage],0,4)) return 0;
        if (stage < 3) {
            snprintf(prefix,sizeof(prefix),"model.visual.layers.norm_%" PRIu32,stage);
            if (!require_tensor(census,prefix,".weight",INKLING_DTYPE_BF16,1,rows[stage],0,0,4)) return 0;
        }
    }
    for (uint32_t layer = 0; layer < 8; layer++) {
        snprintf(prefix,sizeof(prefix),"model.mtp.layers.%" PRIu32 ".",layer);
        if (!require_tensor(census,prefix,"embed_norm.weight",INKLING_DTYPE_BF16,1,4096,0,0,6) ||
            !require_tensor(census,prefix,"hidden_norm.weight",INKLING_DTYPE_BF16,1,4096,0,0,6) ||
            !require_tensor(census,prefix,"input_proj.weight",INKLING_DTYPE_BF16,2,4096,8192,0,6)) return 0;
        snprintf(prefix,sizeof(prefix),"model.mtp.layers.%" PRIu32 ".transformer_block.",layer);
        if (!block_contract(census,prefix,local_layer(mtp_ids,layer),6) || !dense_contract(census,prefix,6)) return 0;
    }
    for (uint64_t i = 0; i < census->index.count; i++) {
        const InklingTensor *tensor = &census->index.entries[i];
        if (!census->seen[i]) {
            fprintf(stderr,"verify: unexpected tensor in required checkpoint families: %s\n",tensor->name);
            return 0;
        }
        if (strncmp(tensor->name,"model.mtp.",10) != 0 &&
            (tensor->dtype == INKLING_DTYPE_BF16 || tensor->dtype == INKLING_DTYPE_F32) &&
            !strstr(tensor->name,".input_amax") && !strstr(tensor->name,".scale2") &&
            !excluded(get(quant,"exclude_modules"),tensor->name)) {
            fprintf(stderr,"verify: missing quantization exclusion for %s\n",tensor->name);
            return 0;
        }
    }
    const InklingJsonValue *modules = get(quant,"exclude_modules");
    for (size_t i = 0; i < inkling_json_array_size(modules); i++) {
        size_t size;
        const char *prefix_name = inkling_json_string(inkling_json_array_at(modules,i),&size);
        int found = strcmp(prefix_name,"model.audio.decoder") == 0;
        for (uint64_t j = 0; !found && j < census->index.count; j++) {
            const char *name = census->index.entries[j].name;
            found = strlen(name) >= size && memcmp(name,prefix_name,size) == 0 &&
                    (name[size] == '\0' || name[size] == '.');
        }
        if (!found) {
            fprintf(stderr,"verify: unknown quantization exclusion %s\n",prefix_name);
            return 0;
        }
    }
    return 1;
}

static int compare_names(const void *a, const void *b)
{
    const InklingTensor *const *left = a, *const *right = b;
    return strcmp((*left)->name, (*right)->name);
}

static int report(const Census *census, int metadata_only)
{
    const InklingTensor **sorted = malloc((size_t)census->index.count * sizeof(*sorted));
    if (!sorted) return 0;
    for (uint64_t i = 0; i < census->index.count; i++) sorted[i] = &census->index.entries[i];
    qsort(sorted,(size_t)census->index.count,sizeof(*sorted),compare_names);
    printf("checkpoint: %s\nsupported metadata revision: %s\nmode: %s\n",MODEL,REVISION,
           metadata_only ? "metadata-only fixtures (full files NOT verified)" : "actual shard lengths; headers only");
    puts("Identity is a metadata contract, not payload authentication.");
    if (census->skip_mtp) puts("MTP: skipped (optional file absent; --with-mtp makes it required)");
    uint64_t count = 0, bytes = 0;
    for (size_t i = 0; i < 7; i++) {
        printf("census %s: tensors=%" PRIu64 " bytes=%" PRIu64 "\n",class_names[i],census->counts[i],census->bytes[i]);
        count += census->counts[i]; bytes += census->bytes[i];
    }
    printf("total: tensors=%" PRIu64 " bytes=%" PRIu64 " indexed=%" PRIu64 "\n",count,bytes,census->index.count);
    for (uint64_t i = 0; i < census->index.count; i++) {
        const InklingTensor *tensor = sorted[i];
        if (tensor->dtype == INKLING_DTYPE_UNKNOWN) continue;
        printf("tensor %s dtype=%s shape=[",tensor->name,dtype_names[tensor->dtype]);
        for (uint32_t axis = 0; axis < tensor->rank; axis++) printf("%s%" PRIu64,axis ? "," : "",tensor->shape[axis]);
        printf("] shard=%s data_range=[%" PRIu64 ",%" PRIu64 ")\n",tensor->shard,tensor->data_offset,
               tensor->data_offset + tensor->byte_length);
    }
    free(sorted);
    puts("verification OK (ranges relative to the shard payload start)");
    return !ferror(stdout);
}

int inkling_verify_model(const char *directory, int with_mtp, int metadata_only)
{
    Census census = {0};
    InklingJsonDocument config = {0}, quant = {0};
    InklingConfig parsed = {0};
    char path[4096], shard[64];
    int ok = 0;
    struct stat status;
    if (!directory || stat(directory,&status) != 0 || !S_ISDIR(status.st_mode)) {
        fputs("verify: MODEL_DIR must be an existing directory\n",stderr);
        goto done;
    }
    if (!path_join(path,sizeof(path),directory,"config.json") || !load_json(path,&config) ||
        !config_contract(config.root)) goto done;
    if (!layer_contract(get(get(config.root,"text_config"),"local_layer_ids"),0) ||
        !layer_contract(get(get(config.root,"mtp_config"),"local_layer_ids"),1)) {
        fputs("verify: local_layer_ids do not match the pinned text/MTP layouts\n",stderr);
        goto done;
    }
    if (!inkling_config_load(path,&parsed)) goto done;
    if (!path_join(path,sizeof(path),directory,"model.safetensors.index.json") || !inkling_index_load(path,&census.index)) {
        fprintf(stderr,"verify: cannot load global index %s\n",path);
        goto done;
    }
    if (!path_join(path,sizeof(path),directory,"hf_quant_config.json") || !load_json(path,&quant) ||
        !quant_contract(get(quant.root,"quantization"))) goto done;
    census.seen = calloc((size_t)census.index.count,1);
    census.names_only = 1;
    if (!census.seen || !tensor_contract(&census,config.root,get(quant.root,"quantization"))) goto done;
    census.names_only = 0;
    if (census.index.count != 1360 || census.index.num_shards != 10 ||
        census.index.total_size != UINT64_C(170733074592)) {
        fputs("verify: index identity mismatch: expected 1360 tensors, 10 shards, total_size=170733074592\n",stderr);
        goto done;
    }
    for (size_t i = 0; i < 10; i++) {
        if (i == 9) strcpy(shard,"mtp.safetensors");
        else snprintf(shard,sizeof(shard),"model-%05zu-of-00009.safetensors",i + 1);
        char relative[80];
        snprintf(relative,sizeof(relative),"%s%s",metadata_only ? "headers/" : "",shard);
        if (!path_join(path,sizeof(path),directory,relative)) goto done;
        if (stat(path,&status) != 0) {
            if (i == 9 && !with_mtp && errno == ENOENT) { census.skip_mtp = 1; continue; }
            fprintf(stderr,"verify: required shard %s: %s\n",path,strerror(errno));
            goto done;
        }
        if (!S_ISREG(status.st_mode)) {
            fprintf(stderr,"verify: shard is not a regular file: %s\n",path);
            goto done;
        }
        int bound;
        if (metadata_only) {
            char *header = NULL;
            uint64_t size = 0;
            bound = inkling_safetensors_read_header(path,&header,&size);
            if (bound) bound = inkling_index_bind_header(&census.index,shard,header,size,8 + size + payload_sizes[i]);
            free(header);
        } else bound = inkling_index_load_shard(&census.index,shard,path);
        if (!bound) {
            fprintf(stderr,"verify: invalid shard %s: check header dtype/shape/ranges and global index names/mapping\n",path);
            goto done;
        }
    }
    if (!tensor_contract(&census,config.root,get(quant.root,"quantization"))) goto done;
    uint64_t bytes = 0;
    for (size_t i = 0; i < 7; i++) bytes += census.bytes[i];
    uint64_t expected = census.index.total_size - (census.skip_mtp ? payload_sizes[9] : 0);
    if (bytes != expected) {
        fprintf(stderr,"verify: census/index byte mismatch: got %" PRIu64 ", expected %" PRIu64 "\n",bytes,expected);
        goto done;
    }
    ok = report(&census,metadata_only);
done:
    free(census.seen);
    inkling_index_free(&census.index);
    inkling_config_free(&parsed);
    inkling_json_document_free(&config);
    inkling_json_document_free(&quant);
    return ok;
}
