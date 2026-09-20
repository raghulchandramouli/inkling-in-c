#include "inkling_json.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INKLING_JSON_MAX_DEPTH 256

typedef struct {
    char *data;
    size_t length;
} JsonText;

typedef struct {
    char *key;
    size_t key_length;
    InklingJsonValue *value;
} JsonMember;

struct InklingJsonValue {
    InklingJsonType type;
    union {
        int boolean;
        JsonText text;
        struct {
            InklingJsonValue **items;
            size_t count;
            size_t capacity;
        } array;
        struct {
            JsonMember *members;
            size_t count;
            size_t capacity;
        } object;
    } as;
};

typedef struct {
    const char *text;
    size_t length;
    size_t position;
    InklingJsonError *error;
    int failed;
} JsonParser;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} JsonBuffer;

static void free_value(InklingJsonValue *value);

static int parser_fail(JsonParser *parser, const char *format, ...)
{
    if (parser->failed) {
        return 0;
    }

    parser->failed = 1;

    if (parser->error == NULL) {
        return 0;
    }

    parser->error->offset = parser->position;
    parser->error->line = 1;
    parser->error->column = 1;

    for (size_t index = 0;
         index < parser->position && index < parser->length;
         index++) {
        if (parser->text[index] == '\n') {
            parser->error->line++;
            parser->error->column = 1;
        } else {
            parser->error->column++;
        }
    }

    va_list arguments;
    va_start(arguments, format);
    vsnprintf(
        parser->error->message,
        sizeof(parser->error->message),
        format,
        arguments
    );
    va_end(arguments);
    return 0;
}

static void skip_whitespace(JsonParser *parser)
{
    while (parser->position < parser->length) {
        char character = parser->text[parser->position];

        if (character != ' ' &&
            character != '\t' &&
            character != '\n' &&
            character != '\r') {
            return;
        }

        parser->position++;
    }
}

static InklingJsonValue *new_value(
    JsonParser *parser,
    InklingJsonType type
)
{
    InklingJsonValue *value = calloc(1, sizeof(*value));

    if (value == NULL) {
        parser_fail(parser, "out of memory");
        return NULL;
    }

    value->type = type;
    return value;
}

static int reserve_buffer(
    JsonParser *parser,
    JsonBuffer *buffer,
    size_t additional
)
{
    if (additional > SIZE_MAX - buffer->length - 1) {
        return parser_fail(parser, "JSON string is too large");
    }

    size_t needed = buffer->length + additional + 1;

    if (needed <= buffer->capacity) {
        return 1;
    }

    size_t capacity = buffer->capacity == 0 ? 16 : buffer->capacity;

    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }

    char *data = realloc(buffer->data, capacity);

    if (data == NULL) {
        return parser_fail(parser, "out of memory");
    }

    buffer->data = data;
    buffer->capacity = capacity;
    return 1;
}

static int append_bytes(
    JsonParser *parser,
    JsonBuffer *buffer,
    const char *bytes,
    size_t count
)
{
    if (!reserve_buffer(parser, buffer, count)) {
        return 0;
    }

    memcpy(buffer->data + buffer->length, bytes, count);
    buffer->length += count;
    buffer->data[buffer->length] = '\0';
    return 1;
}

static int append_codepoint(
    JsonParser *parser,
    JsonBuffer *buffer,
    uint32_t codepoint
)
{
    char bytes[4];
    size_t count = 0;

    if (codepoint <= 0x7f) {
        bytes[0] = (char)codepoint;
        count = 1;
    } else if (codepoint <= 0x7ff) {
        bytes[0] = (char)(0xc0 | (codepoint >> 6));
        bytes[1] = (char)(0x80 | (codepoint & 0x3f));
        count = 2;
    } else if (codepoint <= 0xffff) {
        bytes[0] = (char)(0xe0 | (codepoint >> 12));
        bytes[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[2] = (char)(0x80 | (codepoint & 0x3f));
        count = 3;
    } else if (codepoint <= 0x10ffff) {
        bytes[0] = (char)(0xf0 | (codepoint >> 18));
        bytes[1] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
        bytes[2] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[3] = (char)(0x80 | (codepoint & 0x3f));
        count = 4;
    } else {
        return parser_fail(parser, "invalid Unicode code point");
    }

    return append_bytes(parser, buffer, bytes, count);
}

static int hex_digit(char character, uint32_t *value)
{
    if (character >= '0' && character <= '9') {
        *value = (uint32_t)(character - '0');
        return 1;
    }
    if (character >= 'a' && character <= 'f') {
        *value = (uint32_t)(character - 'a' + 10);
        return 1;
    }
    if (character >= 'A' && character <= 'F') {
        *value = (uint32_t)(character - 'A' + 10);
        return 1;
    }
    return 0;
}

static int parse_hex4(JsonParser *parser, uint32_t *value)
{
    if (parser->length - parser->position < 4) {
        return parser_fail(parser, "incomplete Unicode escape");
    }

    uint32_t result = 0;

    for (unsigned int index = 0; index < 4; index++) {
        uint32_t digit = 0;

        if (!hex_digit(parser->text[parser->position], &digit)) {
            return parser_fail(parser, "invalid Unicode escape");
        }

        result = result * 16 + digit;
        parser->position++;
    }

    *value = result;
    return 1;
}

static size_t utf8_sequence_length(
    const unsigned char *bytes,
    size_t remaining
)
{
    unsigned char first = bytes[0];

    if (first < 0x80) {
        return 1;
    }

    if (first >= 0xc2 && first <= 0xdf) {
        if (remaining >= 2 &&
            bytes[1] >= 0x80 && bytes[1] <= 0xbf) {
            return 2;
        }
        return 0;
    }

    if (first >= 0xe0 && first <= 0xef) {
        if (remaining < 3 ||
            bytes[1] < 0x80 || bytes[1] > 0xbf ||
            bytes[2] < 0x80 || bytes[2] > 0xbf) {
            return 0;
        }
        if ((first == 0xe0 && bytes[1] < 0xa0) ||
            (first == 0xed && bytes[1] > 0x9f)) {
            return 0;
        }
        return 3;
    }

    if (first >= 0xf0 && first <= 0xf4) {
        if (remaining < 4 ||
            bytes[1] < 0x80 || bytes[1] > 0xbf ||
            bytes[2] < 0x80 || bytes[2] > 0xbf ||
            bytes[3] < 0x80 || bytes[3] > 0xbf) {
            return 0;
        }
        if ((first == 0xf0 && bytes[1] < 0x90) ||
            (first == 0xf4 && bytes[1] > 0x8f)) {
            return 0;
        }
        return 4;
    }

    return 0;
}

static int parse_string(JsonParser *parser, JsonText *result)
{
    if (parser->position >= parser->length ||
        parser->text[parser->position] != '"') {
        return parser_fail(parser, "expected string");
    }

    parser->position++;
    JsonBuffer buffer = {0};

    while (parser->position < parser->length) {
        unsigned char character =
            (unsigned char)parser->text[parser->position++];

        if (character == '"') {
            if (!reserve_buffer(parser, &buffer, 0)) {
                free(buffer.data);
                return 0;
            }
            buffer.data[buffer.length] = '\0';
            result->data = buffer.data;
            result->length = buffer.length;
            return 1;
        }

        if (character == '\\') {
            if (parser->position >= parser->length) {
                free(buffer.data);
                return parser_fail(parser, "incomplete string escape");
            }

            char escape = parser->text[parser->position++];
            char decoded = '\0';

            switch (escape) {
                case '"': decoded = '"'; break;
                case '\\': decoded = '\\'; break;
                case '/': decoded = '/'; break;
                case 'b': decoded = '\b'; break;
                case 'f': decoded = '\f'; break;
                case 'n': decoded = '\n'; break;
                case 'r': decoded = '\r'; break;
                case 't': decoded = '\t'; break;
                case 'u': {
                    uint32_t codepoint = 0;

                    if (!parse_hex4(parser, &codepoint)) {
                        free(buffer.data);
                        return 0;
                    }

                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (parser->length - parser->position < 2 ||
                            parser->text[parser->position] != '\\' ||
                            parser->text[parser->position + 1] != 'u') {
                            free(buffer.data);
                            return parser_fail(
                                parser,
                                "high surrogate has no low surrogate"
                            );
                        }

                        parser->position += 2;
                        uint32_t low = 0;

                        if (!parse_hex4(parser, &low)) {
                            free(buffer.data);
                            return 0;
                        }
                        if (low < 0xdc00 || low > 0xdfff) {
                            free(buffer.data);
                            return parser_fail(
                                parser,
                                "invalid low surrogate"
                            );
                        }

                        codepoint = 0x10000 +
                            ((codepoint - 0xd800) << 10) +
                            (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        free(buffer.data);
                        return parser_fail(parser, "unexpected low surrogate");
                    }

                    if (!append_codepoint(parser, &buffer, codepoint)) {
                        free(buffer.data);
                        return 0;
                    }
                    continue;
                }
                default:
                    free(buffer.data);
                    return parser_fail(parser, "invalid string escape");
            }

            if (!append_bytes(parser, &buffer, &decoded, 1)) {
                free(buffer.data);
                return 0;
            }
            continue;
        }

        if (character < 0x20) {
            free(buffer.data);
            return parser_fail(parser, "unescaped control character in string");
        }

        if (character < 0x80) {
            char byte = (char)character;
            if (!append_bytes(parser, &buffer, &byte, 1)) {
                free(buffer.data);
                return 0;
            }
            continue;
        }

        parser->position--;
        const unsigned char *start =
            (const unsigned char *)parser->text + parser->position;
        size_t count = utf8_sequence_length(
            start,
            parser->length - parser->position
        );

        if (count == 0) {
            free(buffer.data);
            return parser_fail(parser, "invalid UTF-8 in string");
        }

        if (!append_bytes(
                parser,
                &buffer,
                parser->text + parser->position,
                count)) {
            free(buffer.data);
            return 0;
        }
        parser->position += count;
    }

    free(buffer.data);
    return parser_fail(parser, "unterminated string");
}

static int parse_value(
    JsonParser *parser,
    unsigned int depth,
    InklingJsonValue **result
);

static int append_array_item(
    JsonParser *parser,
    InklingJsonValue *array,
    InklingJsonValue *item
)
{
    if (array->as.array.count == array->as.array.capacity) {
        size_t capacity = array->as.array.capacity == 0
            ? 8
            : array->as.array.capacity * 2;

        if (capacity < array->as.array.capacity ||
            capacity > SIZE_MAX / sizeof(*array->as.array.items)) {
            return parser_fail(parser, "JSON array is too large");
        }

        InklingJsonValue **items = realloc(
            array->as.array.items,
            capacity * sizeof(*items)
        );

        if (items == NULL) {
            return parser_fail(parser, "out of memory");
        }

        array->as.array.items = items;
        array->as.array.capacity = capacity;
    }

    array->as.array.items[array->as.array.count++] = item;
    return 1;
}

static int parse_array(
    JsonParser *parser,
    unsigned int depth,
    InklingJsonValue **result
)
{
    InklingJsonValue *array = new_value(parser, INKLING_JSON_ARRAY);

    if (array == NULL) {
        return 0;
    }

    parser->position++;
    skip_whitespace(parser);

    if (parser->position < parser->length &&
        parser->text[parser->position] == ']') {
        parser->position++;
        *result = array;
        return 1;
    }

    for (;;) {
        InklingJsonValue *item = NULL;

        if (!parse_value(parser, depth + 1, &item) ||
            !append_array_item(parser, array, item)) {
            free_value(item);
            free_value(array);
            return 0;
        }

        skip_whitespace(parser);

        if (parser->position >= parser->length) {
            free_value(array);
            return parser_fail(parser, "unterminated array");
        }

        char character = parser->text[parser->position++];

        if (character == ']') {
            *result = array;
            return 1;
        }
        if (character != ',') {
            free_value(array);
            return parser_fail(parser, "expected ',' or ']' in array");
        }

        skip_whitespace(parser);
    }
}

static int object_has_key(
    const InklingJsonValue *object,
    const JsonText *key
)
{
    for (size_t index = 0; index < object->as.object.count; index++) {
        const JsonMember *member = &object->as.object.members[index];

        if (member->key_length == key->length &&
            memcmp(member->key, key->data, key->length) == 0) {
            return 1;
        }
    }

    return 0;
}

static int append_object_member(
    JsonParser *parser,
    InklingJsonValue *object,
    JsonText key,
    InklingJsonValue *value
)
{
    if (object->as.object.count == object->as.object.capacity) {
        size_t capacity = object->as.object.capacity == 0
            ? 8
            : object->as.object.capacity * 2;

        if (capacity < object->as.object.capacity ||
            capacity > SIZE_MAX / sizeof(*object->as.object.members)) {
            return parser_fail(parser, "JSON object is too large");
        }

        JsonMember *members = realloc(
            object->as.object.members,
            capacity * sizeof(*members)
        );

        if (members == NULL) {
            return parser_fail(parser, "out of memory");
        }

        object->as.object.members = members;
        object->as.object.capacity = capacity;
    }

    JsonMember *member =
        &object->as.object.members[object->as.object.count++];
    member->key = key.data;
    member->key_length = key.length;
    member->value = value;
    return 1;
}

static int parse_object(
    JsonParser *parser,
    unsigned int depth,
    InklingJsonValue **result
)
{
    InklingJsonValue *object = new_value(parser, INKLING_JSON_OBJECT);

    if (object == NULL) {
        return 0;
    }

    parser->position++;
    skip_whitespace(parser);

    if (parser->position < parser->length &&
        parser->text[parser->position] == '}') {
        parser->position++;
        *result = object;
        return 1;
    }

    for (;;) {
        JsonText key = {0};

        if (!parse_string(parser, &key)) {
            free_value(object);
            return 0;
        }

        if (object_has_key(object, &key)) {
            free(key.data);
            free_value(object);
            return parser_fail(parser, "duplicate object key");
        }

        skip_whitespace(parser);

        if (parser->position >= parser->length ||
            parser->text[parser->position] != ':') {
            free(key.data);
            free_value(object);
            return parser_fail(parser, "expected ':' after object key");
        }

        parser->position++;
        InklingJsonValue *value = NULL;

        if (!parse_value(parser, depth + 1, &value) ||
            !append_object_member(parser, object, key, value)) {
            free(key.data);
            free_value(value);
            free_value(object);
            return 0;
        }

        skip_whitespace(parser);

        if (parser->position >= parser->length) {
            free_value(object);
            return parser_fail(parser, "unterminated object");
        }

        char character = parser->text[parser->position++];

        if (character == '}') {
            *result = object;
            return 1;
        }
        if (character != ',') {
            free_value(object);
            return parser_fail(parser, "expected ',' or '}' in object");
        }

        skip_whitespace(parser);
    }
}

static int parse_number(JsonParser *parser, InklingJsonValue **result)
{
    size_t start = parser->position;

    if (parser->text[parser->position] == '-') {
        parser->position++;
        if (parser->position >= parser->length) {
            return parser_fail(parser, "incomplete number");
        }
    }

    if (parser->text[parser->position] == '0') {
        parser->position++;
        if (parser->position < parser->length &&
            parser->text[parser->position] >= '0' &&
            parser->text[parser->position] <= '9') {
            return parser_fail(parser, "leading zero in number");
        }
    } else if (parser->text[parser->position] >= '1' &&
               parser->text[parser->position] <= '9') {
        do {
            parser->position++;
        } while (parser->position < parser->length &&
                 parser->text[parser->position] >= '0' &&
                 parser->text[parser->position] <= '9');
    } else {
        return parser_fail(parser, "invalid number");
    }

    if (parser->position < parser->length &&
        parser->text[parser->position] == '.') {
        parser->position++;
        size_t fraction_start = parser->position;

        while (parser->position < parser->length &&
               parser->text[parser->position] >= '0' &&
               parser->text[parser->position] <= '9') {
            parser->position++;
        }

        if (parser->position == fraction_start) {
            return parser_fail(parser, "fraction has no digits");
        }
    }

    if (parser->position < parser->length &&
        (parser->text[parser->position] == 'e' ||
         parser->text[parser->position] == 'E')) {
        parser->position++;

        if (parser->position < parser->length &&
            (parser->text[parser->position] == '+' ||
             parser->text[parser->position] == '-')) {
            parser->position++;
        }

        size_t exponent_start = parser->position;

        while (parser->position < parser->length &&
               parser->text[parser->position] >= '0' &&
               parser->text[parser->position] <= '9') {
            parser->position++;
        }

        if (parser->position == exponent_start) {
            return parser_fail(parser, "exponent has no digits");
        }
    }

    size_t length = parser->position - start;
    InklingJsonValue *number = new_value(parser, INKLING_JSON_NUMBER);

    if (number == NULL) {
        return 0;
    }

    if (length == SIZE_MAX) {
        free_value(number);
        return parser_fail(parser, "number is too large");
    }

    number->as.text.data = malloc(length + 1);

    if (number->as.text.data == NULL) {
        free_value(number);
        return parser_fail(parser, "out of memory");
    }

    memcpy(number->as.text.data, parser->text + start, length);
    number->as.text.data[length] = '\0';
    number->as.text.length = length;
    *result = number;
    return 1;
}

static int parse_literal(
    JsonParser *parser,
    const char *literal,
    InklingJsonType type,
    int boolean,
    InklingJsonValue **result
)
{
    size_t length = strlen(literal);

    if (parser->length - parser->position < length ||
        memcmp(parser->text + parser->position, literal, length) != 0) {
        return parser_fail(parser, "invalid JSON literal");
    }

    InklingJsonValue *value = new_value(parser, type);

    if (value == NULL) {
        return 0;
    }

    parser->position += length;
    value->as.boolean = boolean;
    *result = value;
    return 1;
}

static int parse_value(
    JsonParser *parser,
    unsigned int depth,
    InklingJsonValue **result
)
{
    skip_whitespace(parser);

    if (depth > INKLING_JSON_MAX_DEPTH) {
        return parser_fail(parser, "JSON nesting exceeds limit");
    }
    if (parser->position >= parser->length) {
        return parser_fail(parser, "expected JSON value");
    }

    char character = parser->text[parser->position];

    if (character == '{') {
        return parse_object(parser, depth, result);
    }
    if (character == '[') {
        return parse_array(parser, depth, result);
    }
    if (character == '"') {
        InklingJsonValue *string = new_value(parser, INKLING_JSON_STRING);

        if (string == NULL) {
            return 0;
        }
        if (!parse_string(parser, &string->as.text)) {
            free_value(string);
            return 0;
        }
        *result = string;
        return 1;
    }
    if (character == 't') {
        return parse_literal(
            parser,
            "true",
            INKLING_JSON_BOOLEAN,
            1,
            result
        );
    }
    if (character == 'f') {
        return parse_literal(
            parser,
            "false",
            INKLING_JSON_BOOLEAN,
            0,
            result
        );
    }
    if (character == 'n') {
        return parse_literal(
            parser,
            "null",
            INKLING_JSON_NULL,
            0,
            result
        );
    }
    if (character == '-' ||
        (character >= '0' && character <= '9')) {
        return parse_number(parser, result);
    }

    return parser_fail(parser, "unexpected character while parsing value");
}

static void free_value(InklingJsonValue *value)
{
    if (value == NULL) {
        return;
    }

    if (value->type == INKLING_JSON_STRING ||
        value->type == INKLING_JSON_NUMBER) {
        free(value->as.text.data);
    } else if (value->type == INKLING_JSON_ARRAY) {
        for (size_t index = 0; index < value->as.array.count; index++) {
            free_value(value->as.array.items[index]);
        }
        free(value->as.array.items);
    } else if (value->type == INKLING_JSON_OBJECT) {
        for (size_t index = 0; index < value->as.object.count; index++) {
            JsonMember *member = &value->as.object.members[index];
            free(member->key);
            free_value(member->value);
        }
        free(value->as.object.members);
    }

    free(value);
}

int inkling_json_parse(
    const char *text,
    size_t length,
    InklingJsonDocument *document,
    InklingJsonError *error
)
{
    if (document == NULL) {
        return 0;
    }

    document->root = NULL;

    if (error != NULL) {
        error->offset = 0;
        error->line = 1;
        error->column = 1;
        error->message[0] = '\0';
    }

    if (text == NULL) {
        if (error != NULL) {
            snprintf(error->message, sizeof(error->message), "JSON text is NULL");
        }
        return 0;
    }

    JsonParser parser = {
        .text = text,
        .length = length,
        .position = 0,
        .error = error,
        .failed = 0
    };

    InklingJsonValue *root = NULL;

    if (!parse_value(&parser, 0, &root)) {
        free_value(root);
        return 0;
    }

    skip_whitespace(&parser);

    if (parser.position != parser.length) {
        free_value(root);
        return parser_fail(&parser, "trailing content after JSON value");
    }

    document->root = root;
    return 1;
}

void inkling_json_document_free(InklingJsonDocument *document)
{
    if (document == NULL) {
        return;
    }

    free_value(document->root);
    document->root = NULL;
}

InklingJsonType inkling_json_type(const InklingJsonValue *value)
{
    return value == NULL ? INKLING_JSON_NULL : value->type;
}

int inkling_json_boolean(const InklingJsonValue *value, int *result)
{
    if (value == NULL ||
        result == NULL ||
        value->type != INKLING_JSON_BOOLEAN) {
        return 0;
    }

    *result = value->as.boolean;
    return 1;
}

const char *inkling_json_number_text(
    const InklingJsonValue *value,
    size_t *length
)
{
    if (value == NULL || value->type != INKLING_JSON_NUMBER) {
        return NULL;
    }

    if (length != NULL) {
        *length = value->as.text.length;
    }
    return value->as.text.data;
}

int inkling_json_number_u64(
    const InklingJsonValue *value,
    uint64_t *result
)
{
    if (value == NULL ||
        result == NULL ||
        value->type != INKLING_JSON_NUMBER ||
        value->as.text.length == 0 ||
        value->as.text.data[0] == '-') {
        return 0;
    }

    uint64_t number = 0;

    for (size_t index = 0; index < value->as.text.length; index++) {
        char character = value->as.text.data[index];

        if (character < '0' || character > '9') {
            return 0;
        }

        uint64_t digit = (uint64_t)(character - '0');

        if (number > (UINT64_MAX - digit) / 10) {
            return 0;
        }
        number = number * 10 + digit;
    }

    *result = number;
    return 1;
}

int inkling_json_number_i64(
    const InklingJsonValue *value,
    int64_t *result
)
{
    if (value == NULL ||
        result == NULL ||
        value->type != INKLING_JSON_NUMBER ||
        value->as.text.length == 0) {
        return 0;
    }

    int negative = value->as.text.data[0] == '-';
    size_t index = negative ? 1 : 0;
    uint64_t limit = negative
        ? (uint64_t)INT64_MAX + 1
        : (uint64_t)INT64_MAX;
    uint64_t magnitude = 0;

    if (index == value->as.text.length) {
        return 0;
    }

    for (; index < value->as.text.length; index++) {
        char character = value->as.text.data[index];

        if (character < '0' || character > '9') {
            return 0;
        }

        uint64_t digit = (uint64_t)(character - '0');

        if (magnitude > (limit - digit) / 10) {
            return 0;
        }
        magnitude = magnitude * 10 + digit;
    }

    if (negative) {
        *result = magnitude == (uint64_t)INT64_MAX + 1
            ? INT64_MIN
            : -(int64_t)magnitude;
    } else {
        *result = (int64_t)magnitude;
    }
    return 1;
}

int inkling_json_number_double(
    const InklingJsonValue *value,
    double *result
)
{
    if (value == NULL ||
        result == NULL ||
        value->type != INKLING_JSON_NUMBER) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double number = strtod(value->as.text.data, &end);

    if (errno == ERANGE ||
        end != value->as.text.data + value->as.text.length ||
        !isfinite(number)) {
        return 0;
    }

    *result = number;
    return 1;
}

const char *inkling_json_string(
    const InklingJsonValue *value,
    size_t *length
)
{
    if (value == NULL || value->type != INKLING_JSON_STRING) {
        return NULL;
    }

    if (length != NULL) {
        *length = value->as.text.length;
    }
    return value->as.text.data;
}

size_t inkling_json_array_size(const InklingJsonValue *value)
{
    return value != NULL && value->type == INKLING_JSON_ARRAY
        ? value->as.array.count
        : 0;
}

const InklingJsonValue *inkling_json_array_at(
    const InklingJsonValue *value,
    size_t index
)
{
    if (value == NULL ||
        value->type != INKLING_JSON_ARRAY ||
        index >= value->as.array.count) {
        return NULL;
    }

    return value->as.array.items[index];
}

size_t inkling_json_object_size(const InklingJsonValue *value)
{
    return value != NULL && value->type == INKLING_JSON_OBJECT
        ? value->as.object.count
        : 0;
}

const InklingJsonValue *inkling_json_object_get(
    const InklingJsonValue *value,
    const char *key
)
{
    if (value == NULL ||
        value->type != INKLING_JSON_OBJECT ||
        key == NULL) {
        return NULL;
    }

    size_t length = strlen(key);

    for (size_t index = 0; index < value->as.object.count; index++) {
        const JsonMember *member = &value->as.object.members[index];

        if (member->key_length == length &&
            memcmp(member->key, key, length) == 0) {
            return member->value;
        }
    }

    return NULL;
}

const char *inkling_json_object_key_at(
    const InklingJsonValue *value,
    size_t index,
    size_t *length
)
{
    if (value == NULL ||
        value->type != INKLING_JSON_OBJECT ||
        index >= value->as.object.count) {
        return NULL;
    }

    const JsonMember *member = &value->as.object.members[index];

    if (length != NULL) {
        *length = member->key_length;
    }
    return member->key;
}

const InklingJsonValue *inkling_json_object_value_at(
    const InklingJsonValue *value,
    size_t index
)
{
    if (value == NULL ||
        value->type != INKLING_JSON_OBJECT ||
        index >= value->as.object.count) {
        return NULL;
    }

    return value->as.object.members[index].value;
}
