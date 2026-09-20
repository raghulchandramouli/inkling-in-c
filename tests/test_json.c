#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/io/inkling_json.h"

static int failures = 0;

static void expect(int condition, const char *what)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

static int parse(
    const char *text,
    InklingJsonDocument *document,
    InklingJsonError *error
)
{
    return inkling_json_parse(text, strlen(text), document, error);
}

static void expect_invalid(const char *text, const char *what)
{
    InklingJsonDocument document = {0};
    InklingJsonError error;

    if (parse(text, &document, &error)) {
        fprintf(stderr, "FAIL: accepted %s\n", what);
        failures++;
    } else {
        expect(error.message[0] != '\0', "parse error has a message");
        expect(error.line >= 1, "parse error has a line");
        expect(error.column >= 1, "parse error has a column");
    }

    inkling_json_document_free(&document);
}

static void test_document(void)
{
    const char *text =
        "{"
        "\"name\":\"Inkling \\uD83D\\uDE00\","
        "\"enabled\":true,"
        "\"nothing\":null,"
        "\"values\":[0,-42,1.25e2],"
        "\"nested\":{\"escaped\":\"a\\u0000b\"}"
        "}";
    InklingJsonDocument document = {0};
    InklingJsonError error;

    expect(parse(text, &document, &error), "valid nested document parses");

    const InklingJsonValue *root = document.root;
    expect(inkling_json_type(root) == INKLING_JSON_OBJECT, "root is object");
    expect(inkling_json_object_size(root) == 5, "object member count");

    const InklingJsonValue *name = inkling_json_object_get(root, "name");
    size_t length = 0;
    const char *name_text = inkling_json_string(name, &length);
    const char expected_name[] = "Inkling \xf0\x9f\x98\x80";

    expect(name_text != NULL, "string getter accepts string");
    expect(
        length == sizeof(expected_name) - 1 &&
        memcmp(name_text, expected_name, length) == 0,
        "surrogate pair decodes to UTF-8"
    );

    int enabled = 0;
    expect(
        inkling_json_boolean(
            inkling_json_object_get(root, "enabled"),
            &enabled
        ) && enabled,
        "boolean getter"
    );
    expect(
        inkling_json_type(inkling_json_object_get(root, "nothing")) ==
            INKLING_JSON_NULL,
        "null value"
    );

    const InklingJsonValue *values =
        inkling_json_object_get(root, "values");
    expect(inkling_json_array_size(values) == 3, "array size");

    uint64_t unsigned_number = 1;
    int64_t signed_number = 0;
    double real_number = 0.0;

    expect(
        inkling_json_number_u64(
            inkling_json_array_at(values, 0),
            &unsigned_number
        ) && unsigned_number == 0,
        "unsigned integer conversion"
    );
    expect(
        inkling_json_number_i64(
            inkling_json_array_at(values, 1),
            &signed_number
        ) && signed_number == -42,
        "signed integer conversion"
    );
    expect(
        inkling_json_number_double(
            inkling_json_array_at(values, 2),
            &real_number
        ) && real_number == 125.0,
        "floating-point conversion"
    );

    const InklingJsonValue *nested =
        inkling_json_object_get(root, "nested");
    const char *escaped = inkling_json_string(
        inkling_json_object_get(nested, "escaped"),
        &length
    );
    const char expected_escaped[] = {'a', '\0', 'b'};

    expect(
        escaped != NULL &&
        length == sizeof(expected_escaped) &&
        memcmp(escaped, expected_escaped, length) == 0,
        "embedded NUL preserves string length"
    );

    size_t key_length = 0;
    const char *first_key =
        inkling_json_object_key_at(root, 0, &key_length);
    expect(
        first_key != NULL &&
        key_length == 4 &&
        memcmp(first_key, "name", 4) == 0,
        "object iteration returns key"
    );
    expect(
        inkling_json_object_value_at(root, 0) == name,
        "object iteration returns value"
    );
    expect(
        inkling_json_object_get(root, "missing") == NULL,
        "missing object key"
    );
    expect(
        inkling_json_array_at(values, 3) == NULL,
        "array bounds check"
    );

    inkling_json_document_free(&document);
    expect(document.root == NULL, "document free clears root");
    inkling_json_document_free(&document);
}

static void test_number_bounds(void)
{
    const char *text =
        "[18446744073709551615,18446744073709551616,"
        "-9223372036854775808,-9223372036854775809,1.5,1e999]";
    InklingJsonDocument document = {0};
    InklingJsonError error;

    expect(parse(text, &document, &error), "number bounds document parses");

    uint64_t unsigned_number = 0;
    int64_t signed_number = 0;
    double real_number = 0.0;

    expect(
        inkling_json_number_u64(
            inkling_json_array_at(document.root, 0),
            &unsigned_number
        ) && unsigned_number == UINT64_MAX,
        "UINT64_MAX accepted"
    );
    expect(
        !inkling_json_number_u64(
            inkling_json_array_at(document.root, 1),
            &unsigned_number
        ),
        "uint64 overflow rejected"
    );
    expect(
        inkling_json_number_i64(
            inkling_json_array_at(document.root, 2),
            &signed_number
        ) && signed_number == INT64_MIN,
        "INT64_MIN accepted"
    );
    expect(
        !inkling_json_number_i64(
            inkling_json_array_at(document.root, 3),
            &signed_number
        ),
        "int64 overflow rejected"
    );
    expect(
        !inkling_json_number_i64(
            inkling_json_array_at(document.root, 4),
            &signed_number
        ),
        "fraction rejected as integer"
    );
    expect(
        !inkling_json_number_double(
            inkling_json_array_at(document.root, 5),
            &real_number
        ),
        "double overflow rejected"
    );

    size_t length = 0;
    const char *number_text = inkling_json_number_text(
        inkling_json_array_at(document.root, 2),
        &length
    );
    expect(
        number_text != NULL &&
        length == 20 &&
        memcmp(number_text, "-9223372036854775808", length) == 0,
        "exact number text retained"
    );

    inkling_json_document_free(&document);
}

static void test_invalid_documents(void)
{
    static const char *cases[] = {
        "",
        " ",
        "tru",
        "true false",
        "01",
        "-",
        ".1",
        "1.",
        "1e",
        "1e+",
        "[1,]",
        "[1 2]",
        "{\"a\":1,}",
        "{\"a\" 1}",
        "{\"a\":1 \"b\":2}",
        "{\"a\":1,\"a\":2}",
        "{\"a\":1,\"\\u0061\":2}",
        "\"\\x\"",
        "\"\\u12xz\"",
        "\"\\uD800\"",
        "\"\\uDC00\"",
        "\"unterminated",
        "nul",
        "{}x"
    };

    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        expect_invalid(cases[index], cases[index]);
    }

    const char control[] = {'"', 'a', '\n', 'b', '"'};
    InklingJsonDocument document = {0};
    InklingJsonError error;
    expect(
        !inkling_json_parse(
            control,
            sizeof(control),
            &document,
            &error
        ),
        "unescaped control character rejected"
    );
    inkling_json_document_free(&document);

    const char invalid_utf8[] = {'"', (char)0xc0, (char)0x80, '"'};
    expect(
        !inkling_json_parse(
            invalid_utf8,
            sizeof(invalid_utf8),
            &document,
            &error
        ),
        "invalid UTF-8 rejected"
    );
    inkling_json_document_free(&document);

    expect(
        !inkling_json_parse(NULL, 0, &document, &error),
        "NULL input rejected"
    );
    expect(
        !inkling_json_parse("null", 4, NULL, &error),
        "NULL document rejected"
    );
}

static void test_error_location(void)
{
    const char *text = "{\n  \"valid\": true,\n  }";
    InklingJsonDocument document = {0};
    InklingJsonError error;

    expect(!parse(text, &document, &error), "multiline error rejected");
    expect(error.line == 3, "error line reported");
    expect(error.column == 3, "error column reported");
    inkling_json_document_free(&document);
}

static void test_depth_limit(void)
{
    const size_t depth = 258;
    const size_t length = depth * 2 + 1;
    char *text = malloc(length + 1);

    if (text == NULL) {
        expect(0, "allocate depth test");
        return;
    }

    memset(text, '[', depth);
    text[depth] = '0';
    memset(text + depth + 1, ']', depth);
    text[length] = '\0';

    InklingJsonDocument document = {0};
    InklingJsonError error;
    expect(!parse(text, &document, &error), "nesting limit enforced");
    inkling_json_document_free(&document);
    free(text);
}

static char *read_file(const char *path, size_t *length)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL ||
        fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return NULL;
    }

    long file_length = ftell(file);

    if (file_length < 0 ||
        (uintmax_t)file_length >= (uintmax_t)SIZE_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    size_t size = (size_t)file_length;
    char *text = malloc(size + 1);

    if (text == NULL) {
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(text, 1, size, file);
    fclose(file);

    if (bytes_read != size) {
        free(text);
        return NULL;
    }

    text[size] = '\0';
    *length = size;
    return text;
}

static void test_checkpoint_config(const char *path)
{
    size_t length = 0;
    char *text = read_file(path, &length);

    if (text == NULL) {
        expect(0, "read checkpoint config");
        return;
    }

    InklingJsonDocument document = {0};
    InklingJsonError error;

    if (!inkling_json_parse(text, length, &document, &error)) {
        fprintf(
            stderr,
            "FAIL: checkpoint config JSON at %zu:%zu: %s\n",
            error.line,
            error.column,
            error.message
        );
        failures++;
        free(text);
        return;
    }

    const InklingJsonValue *text_config =
        inkling_json_object_get(document.root, "text_config");
    const InklingJsonValue *local_layers =
        inkling_json_object_get(text_config, "local_layer_ids");
    uint64_t model_max_length = 0;
    uint64_t dense_mlp_idx = 0;

    expect(
        inkling_json_number_u64(
            inkling_json_object_get(text_config, "model_max_length"),
            &model_max_length
        ) && model_max_length == 1048576,
        "real nested config number"
    );
    expect(
        inkling_json_number_u64(
            inkling_json_object_get(text_config, "dense_mlp_idx"),
            &dense_mlp_idx
        ) && dense_mlp_idx == 2,
        "real dense MLP boundary"
    );
    expect(
        inkling_json_array_size(local_layers) == 35,
        "real local layer array"
    );

    inkling_json_document_free(&document);
    free(text);
}

static void test_checkpoint_index(const char *path)
{
    size_t length = 0;
    char *text = read_file(path, &length);

    if (text == NULL) {
        expect(0, "read checkpoint index");
        return;
    }

    InklingJsonDocument document = {0};
    InklingJsonError error;

    if (!inkling_json_parse(text, length, &document, &error)) {
        fprintf(
            stderr,
            "FAIL: checkpoint index JSON at %zu:%zu: %s\n",
            error.line,
            error.column,
            error.message
        );
        failures++;
        free(text);
        return;
    }

    const InklingJsonValue *metadata =
        inkling_json_object_get(document.root, "metadata");
    const InklingJsonValue *weight_map =
        inkling_json_object_get(document.root, "weight_map");
    uint64_t total_size = 0;

    expect(
        inkling_json_number_u64(
            inkling_json_object_get(metadata, "total_size"),
            &total_size
        ) && total_size == 170733074592ULL,
        "real index total size"
    );
    expect(
        inkling_json_object_size(weight_map) == 1360,
        "real index tensor count"
    );

    inkling_json_document_free(&document);
    free(text);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <config.json> <index.json>\n", argv[0]);
        return EXIT_FAILURE;
    }

    test_document();
    test_number_bounds();
    test_invalid_documents();
    test_error_location();
    test_depth_limit();
    test_checkpoint_config(argv[1]);
    test_checkpoint_index(argv[2]);

    if (failures != 0) {
        fprintf(stderr, "%d JSON test(s) failed\n", failures);
        return EXIT_FAILURE;
    }

    puts("All JSON tests passed");
    return EXIT_SUCCESS;
}
