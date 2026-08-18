#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inkling/inkling.h"

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

    expect(
        inkling_config_load(path, &config),
        "real config loads"
    );

    expect_u32(config.model_max_length, 1048576, "model_max_length");
    expect_u32(config.vocab_size, 201024, "vocab_size");
    expect_u32(config.eos_token_id, 200006, "eos_token_id");
    expect_u32(config.hidden_size, 4096, "hidden_size");
    expect_u32(config.num_hidden_layers, 42, "num_hidden_layers");
    expect_u32(config.num_attention_heads, 32, "num_attention_heads");
    expect_u32(config.num_key_value_heads, 8, "num_key_value_heads");
    expect_u32(config.head_dim, 128, "head_dim");
    expect_u32(config.sliding_window_size, 512, "sliding_window_size");
    expect_u32(config.global_attention_stride, 6, "global_attention_stride");
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
        .global_attention_stride = 6,
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

static void test_load_failures(void)
{
    InklingConfig config;

    expect(
        !inkling_config_load("/nonexistent/config.json", &config),
        "missing file rejected"
    );

    expect(
        !inkling_config_load(NULL, &config),
        "NULL path rejected"
    );

    expect(
        !inkling_config_load("tests/fixtures/inkling-small-config.json", NULL),
        "NULL config rejected"
    );

    char path[64];
    const char *missing_key = "{\"hidden_size\": 4096}";
    const char *bad_number = "{\"hidden_size\": -4}";
    const char *empty_object = "{}";

    if (write_temp_file(path, sizeof(path), missing_key)) {
        expect(
            !inkling_config_load(path, &config),
            "missing required key rejected"
        );
        unlink(path);
    }

    if (write_temp_file(path, sizeof(path), bad_number)) {
        expect(
            !inkling_config_load(path, &config),
            "negative number rejected"
        );
        unlink(path);
    }

    if (write_temp_file(path, sizeof(path), empty_object)) {
        expect(
            !inkling_config_load(path, &config),
            "empty object rejected"
        );
        unlink(path);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(
            stderr,
            "usage: %s <inkling-small-config.json>\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    test_real_config(argv[1]);
    test_validity();
    test_load_failures();

    if (failures != 0) {
        fprintf(stderr, "%d config test(s) failed\n", failures);
        return EXIT_FAILURE;
    }

    puts("All config tests passed");
    return EXIT_SUCCESS;
}