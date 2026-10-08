#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inkling/inkling.h"

#define CONFIG_FIELDS_BEFORE_HIDDEN \
    "\"model_max_length\":1024," \
    "\"vocab_size\":1000," \
    "\"hidden_size\":"

#define CONFIG_FIELDS_AFTER_HIDDEN \
    ",\"num_hidden_layers\":12," \
    "\"num_attention_heads\":32," \
    "\"num_key_value_heads\":8," \
    "\"head_dim\":128," \
    "\"sliding_window_size\":512," \
    "\"d_rel\":16," \
    "\"rel_extent\":128," \
    "\"sconv_kernel_size\":4," \
    "\"n_routed_experts\":256," \
    "\"num_experts_per_tok\":6," \
    "\"n_shared_experts\":2," \
    "\"dense_intermediate_size\":16384," \
    "\"intermediate_size\":2048," \
    "\"rms_norm_eps\":0.000001," \
    "\"route_scale\":8.0,"

#define VALID_LOCAL_LAYERS \
    "\"local_layer_ids\":[0,1,2,3,4,6,7,8,9,10]"

#define VALID_TEXT_CONFIG \
    CONFIG_FIELDS_BEFORE_HIDDEN "4096" \
    CONFIG_FIELDS_AFTER_HIDDEN VALID_LOCAL_LAYERS

#define VALID_CONFIG \
    "{\"eos_token_id\":900,\"text_config\":{" \
    VALID_TEXT_CONFIG \
    "}}"

static int failures = 0;

static void expect(int condition, const char *what)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static void expect_u32(
    uint32_t actual,
    uint32_t expected,
    const char *what
)
{
    if (actual != expected) {
        fprintf(
            stderr,
            "FAIL: %s = %" PRIu32 ", expected %" PRIu32 "\n",
            what,
            actual,
            expected
        );
        failures++;
    }
}

static void expect_float(
    float actual,
    float expected,
    const char *what
)
{
    if (actual != expected) {
        fprintf(
            stderr,
            "FAIL: %s = %.9g, expected %.9g\n",
            what,
            actual,
            expected
        );
        failures++;
    }
}

static void test_real_config(const char *path)
{
    InklingConfig config;

    int loaded = inkling_config_load(path, &config);
    expect(loaded, "real config loads");
    if (!loaded) {
        return;
    }

    expect_u32(config.model_max_length, 1048576, "model_max_length");
    expect_u32(config.vocab_size, 201024, "vocab_size");
    expect_u32(config.eos_token_id, 200006, "eos_token_id");
    expect_u32(config.hidden_size, 4096, "hidden_size");
    expect_u32(config.num_hidden_layers, 42, "num_hidden_layers");
    expect_u32(config.num_attention_heads, 32, "num_attention_heads");
    expect_u32(config.num_key_value_heads, 8, "num_key_value_heads");
    expect_u32(config.head_dim, 128, "head_dim");
    expect_u32(config.sliding_window_size, 512, "sliding_window_size");
    static const uint32_t local_layers[] = {
        0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,
        24,25,26,27,28,30,31,32,33,34,36,37,38,39,40
    };
    expect(
        config.num_local_layers == sizeof(local_layers) / sizeof(local_layers[0]) &&
        memcmp(config.local_layer_ids, local_layers, sizeof(local_layers)) == 0,
        "exact real local-layer list preserved"
    );
    expect_u32(config.relative_dimension, 16, "relative_dimension");
    expect_u32(config.relative_extent, 1024, "relative_extent");
    expect_u32(config.sconv_kernel_size, 4, "sconv_kernel_size");
    expect_u32(config.num_routed_experts, 256, "num_routed_experts");
    expect_u32(config.num_experts_per_token, 6, "num_experts_per_token");
    expect_u32(config.num_shared_experts, 2, "num_shared_experts");
    expect_u32(config.dense_intermediate_size, 16384, "dense_intermediate_size");
    expect_u32(config.expert_intermediate_size, 2048, "expert_intermediate_size");
    expect_float(config.rms_norm_epsilon, 1e-6f, "rms_norm_epsilon");
    expect_float(config.route_scale, 8.0f, "route_scale");
    inkling_config_free(&config);
    expect(config.local_layer_ids == NULL && config.num_local_layers == 0,
           "config free clears owned list");
    inkling_config_free(&config);
    inkling_config_free(NULL);
}

static void test_validity(void)
{
    InklingConfig config = {
        .model_max_length = 1024,
        .vocab_size = 1000,
        .eos_token_id = 900,
        .hidden_size = 4096,
        .num_hidden_layers = 12,
        .num_attention_heads = 32,
        .num_key_value_heads = 8,
        .head_dim = 128,
        .sliding_window_size = 512,
        .relative_dimension = 16,
        .relative_extent = 128,
        .sconv_kernel_size = 4,
        .num_routed_experts = 256,
        .num_experts_per_token = 6,
        .num_shared_experts = 2,
        .dense_intermediate_size = 16384,
        .expert_intermediate_size = 2048,
        .rms_norm_epsilon = 1e-6f,
        .route_scale = 8.0f
    };

    expect(inkling_config_is_valid(&config), "valid config accepted");
    expect(!inkling_config_is_valid(NULL), "NULL config rejected");

    InklingConfig copy;

    copy = config;
    copy.model_max_length = 0;
    expect(!inkling_config_is_valid(&copy), "zero context rejected");

    copy = config;
    copy.hidden_size = 4095;
    expect(!inkling_config_is_valid(&copy), "hidden/head mismatch rejected");

    copy = config;
    copy.num_attention_heads = 12;
    expect(!inkling_config_is_valid(&copy), "heads not divisible by KV heads rejected");

    copy = config;
    copy.num_experts_per_token = 257;
    expect(!inkling_config_is_valid(&copy), "experts-per-token over routed rejected");

    copy = config;
    copy.eos_token_id = 1000;
    expect(!inkling_config_is_valid(&copy), "eos at vocab size rejected");

    copy = config;
    copy.sliding_window_size = 2048;
    expect(!inkling_config_is_valid(&copy), "window over context rejected");

    copy = config;
    copy.rms_norm_epsilon = 0.0f;
    expect(!inkling_config_is_valid(&copy), "zero epsilon rejected");

    copy = config;
    copy.route_scale = -1.0f;
    expect(!inkling_config_is_valid(&copy), "negative route scale rejected");

    copy = config;
    copy.rms_norm_epsilon = NAN;
    expect(!inkling_config_is_valid(&copy), "NaN epsilon rejected");

    copy = config;
    copy.route_scale = INFINITY;
    expect(!inkling_config_is_valid(&copy), "infinite route scale rejected");

    copy = config;
    copy.num_local_layers = 1;
    expect(!inkling_config_is_valid(&copy), "missing local-layer storage rejected");

    uint32_t local_layers[] = {3, 3};
    copy.local_layer_ids = local_layers;
    copy.num_local_layers = 2;
    expect(!inkling_config_is_valid(&copy), "duplicate layer in public config rejected");
    local_layers[1] = config.num_hidden_layers;
    expect(!inkling_config_is_valid(&copy), "out-of-range layer in public config rejected");
    copy.num_local_layers = (size_t)config.num_hidden_layers + 1;
    expect(!inkling_config_is_valid(&copy), "too many local layers rejected");

    copy = config;
    uint32_t *dimensions[] = {&copy.relative_dimension, &copy.relative_extent,
        &copy.sconv_kernel_size, &copy.dense_intermediate_size, &copy.expert_intermediate_size};
    for (size_t i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); i++) {
        uint32_t original = *dimensions[i];
        *dimensions[i] = 0;
        expect(!inkling_config_is_valid(&copy), "zero model dimension rejected");
        *dimensions[i] = original;
    }
    copy.sliding_window_size = 0;
    expect(inkling_config_is_valid(&copy), "all-global layout needs no sliding window");
    uint32_t local = 0;
    copy.local_layer_ids = &local;
    copy.num_local_layers = 1;
    expect(!inkling_config_is_valid(&copy), "local attention requires a nonzero window");
}

static int write_temp_file(char *path, size_t path_size, const char *content)
{
    snprintf(path, path_size, "/tmp/inkling-test-XXXXXX");

    int descriptor = mkstemp(path);

    if (descriptor < 0) {
        return 0;
    }

    size_t length = strlen(content);

    if (write(descriptor, content, length) != (ssize_t)length) {
        close(descriptor);
        unlink(path);
        return 0;
    }

    close(descriptor);
    return 1;
}

static int load_temp_config(
    const char *content,
    InklingConfig *config
)
{
    char path[64];

    if (!write_temp_file(path, sizeof(path), content)) {
        return 0;
    }

    int loaded = inkling_config_load(path, config);
    unlink(path);
    return loaded;
}

static void test_structured_loading(void)
{
    InklingConfig config = {0};

    expect(
        load_temp_config(VALID_CONFIG, &config),
        "minimal structured config loads"
    );
    expect_u32(config.hidden_size, 4096, "structured hidden_size");
    expect(config.num_local_layers == 10, "structured local-layer count");
    inkling_config_free(&config);

    const char *shadowed =
        "{"
        "\"hidden_size\":1,"
        "\"audio_config\":{\"hidden_size\":2},"
        "\"mtp_config\":{\"local_layer_ids\":[11]},"
        "\"eos_token_id\":900,"
        "\"text_config\":{" VALID_TEXT_CONFIG "}"
        "}";

    expect(
        load_temp_config(shadowed, &config),
        "unrelated repeated field names do not shadow text config"
    );
    expect_u32(config.hidden_size, 4096, "nested text hidden_size wins");
    expect(config.num_local_layers == 10 && config.local_layer_ids[0] == 0,
           "MTP local layers do not shadow text local layers");
    inkling_config_free(&config);
}

static void test_local_layer_lists(void)
{
    static const struct {
        const char *json;
        size_t count;
        uint32_t ids[12];
    } cases[] = {
        {"[0,1,2,3,4,6,7,8,9,11]", 10, {0,1,2,3,4,6,7,8,9,11}},
        {"[11,0,5]", 3, {11,0,5}},
        {"[]", 0, {0}},
        {"[0,1,2,3,4,5,6,7,8,9,10,11]", 12, {0,1,2,3,4,5,6,7,8,9,10,11}}
    };

    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        char json[1024];
        snprintf(json, sizeof(json),
                 "{\"eos_token_id\":900,\"text_config\":{"
                 CONFIG_FIELDS_BEFORE_HIDDEN "4096" CONFIG_FIELDS_AFTER_HIDDEN
                 "\"local_layer_ids\":%s}}", cases[index].json);
        InklingConfig config = {0};
        int loaded = load_temp_config(json, &config);
        expect(loaded, "valid explicit local-layer layout accepted");
        if (loaded) {
            expect(config.num_local_layers == cases[index].count,
                   "explicit local-layer count preserved");
            expect(config.num_local_layers == cases[index].count &&
                   (cases[index].count == 0 ||
                    memcmp(config.local_layer_ids, cases[index].ids,
                           cases[index].count * sizeof(uint32_t)) == 0),
                   "explicit local-layer order and IDs preserved");
        }
        inkling_config_free(&config);
    }

    static const char *invalid[] = {
        "null", "0", "{}", "[\"0\"]", "[true]", "[null]", "[[]]",
        "[-1]", "[0.5]", "[1e0]", "[12]", "[4294967296]",
        "[18446744073709551616]", "[11,0,11]",
        "[0,1,2,3,4,5,6,7,8,9,10,11,0]"
    };
    for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]); index++) {
        char json[1024];
        snprintf(json, sizeof(json),
                 "{\"eos_token_id\":900,\"text_config\":{"
                 CONFIG_FIELDS_BEFORE_HIDDEN "4096" CONFIG_FIELDS_AFTER_HIDDEN
                 "\"local_layer_ids\":%s}}", invalid[index]);
        InklingConfig config = {.hidden_size = 123};
        expect(!load_temp_config(json, &config), "invalid local-layer list rejected");
        expect(config.hidden_size == 123 && config.local_layer_ids == NULL &&
               config.num_local_layers == 0, "failed load leaves config unchanged");
        inkling_config_free(&config);
    }
}

static void test_load_failures(void)
{
    InklingConfig config = {0};

    expect(
        !inkling_config_load("/nonexistent/config.json", &config),
        "missing file rejected"
    );

    expect(
        !inkling_config_load(NULL, &config),
        "NULL path rejected"
    );

    expect(
        !inkling_config_load("tests/fixtures/checkpoint/config.json", NULL),
        "NULL config rejected"
    );

    static const struct {
        const char *content;
        const char *what;
    } cases[] = {
        {
            "{}",
            "missing text config rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":[]}",
            "non-object text config rejected"
        },
        {
            "{\"eos_token_id\":900,\"eos_token_id\":900,"
            "\"text_config\":{" VALID_TEXT_CONFIG "}}",
            "duplicate root key rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":{"
            VALID_TEXT_CONFIG ",\"hidden_size\":4096}}",
            "duplicate text key rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":{"
            CONFIG_FIELDS_BEFORE_HIDDEN "\"4096\""
            CONFIG_FIELDS_AFTER_HIDDEN VALID_LOCAL_LAYERS "}}",
            "string integer rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":{"
            CONFIG_FIELDS_BEFORE_HIDDEN "4294967296"
            CONFIG_FIELDS_AFTER_HIDDEN VALID_LOCAL_LAYERS "}}",
            "uint32 overflow rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":{"
            CONFIG_FIELDS_BEFORE_HIDDEN "4096"
            CONFIG_FIELDS_AFTER_HIDDEN
            "\"local_layer_ids\":[0,1,2,3,4,4,6,7,8,9,10]}}",
            "duplicate local layer rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":{"
            CONFIG_FIELDS_BEFORE_HIDDEN "4096"
            CONFIG_FIELDS_AFTER_HIDDEN
            "\"other_layers\":[0,1,2,3,4,6,7,8,9,11]}}",
            "missing local-layer list rejected"
        },
        {
            "{\"eos_token_id\":900,\"text_config\":{"
            CONFIG_FIELDS_BEFORE_HIDDEN "4096"
            CONFIG_FIELDS_AFTER_HIDDEN
            "\"local_layer_ids\":[0,1,2,3,4,6,7,8,9,12]}}",
            "out-of-range local layer rejected"
        },
        {
            "{\"text_config\":{" VALID_TEXT_CONFIG
            ",\"eos_token_id\":900}}",
            "nested eos token does not satisfy root field"
        },
        {
            VALID_CONFIG " trailing",
            "trailing content rejected"
        }
    };

    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        expect(
            !load_temp_config(cases[index].content, &config),
            cases[index].what
        );
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(
            stderr,
            "usage: %s <config.json>\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    test_real_config(argv[1]);
    test_validity();
    test_structured_loading();
    test_local_layer_lists();
    test_load_failures();

    if (failures != 0) {
        fprintf(stderr, "%d config test(s) failed\n", failures);
        return EXIT_FAILURE;
    }

    puts("All config tests passed");
    return EXIT_SUCCESS;
}
