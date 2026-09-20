#ifndef INKLING_JSON_H
#define INKLING_JSON_H

#include <stddef.h>
#include <stdint.h>

#define INKLING_JSON_ERROR_MESSAGE_SIZE 160

typedef enum {
    INKLING_JSON_NULL = 0,
    INKLING_JSON_BOOLEAN,
    INKLING_JSON_NUMBER,
    INKLING_JSON_STRING,
    INKLING_JSON_ARRAY,
    INKLING_JSON_OBJECT
} InklingJsonType;

typedef struct InklingJsonValue InklingJsonValue;

typedef struct {
    InklingJsonValue *root;
} InklingJsonDocument;

typedef struct {
    size_t offset;
    size_t line;
    size_t column;
    char message[INKLING_JSON_ERROR_MESSAGE_SIZE];
} InklingJsonError;

/*
 * Parse exactly length bytes of UTF-8 JSON. The document owns the resulting
 * tree and must be released with inkling_json_document_free(). On failure the
 * document is empty and error, when non-NULL, describes the first error.
 */
int inkling_json_parse(
    const char *text,
    size_t length,
    InklingJsonDocument *document,
    InklingJsonError *error
);

void inkling_json_document_free(InklingJsonDocument *document);

InklingJsonType inkling_json_type(const InklingJsonValue *value);

int inkling_json_boolean(
    const InklingJsonValue *value,
    int *result
);

int inkling_json_number_u64(
    const InklingJsonValue *value,
    uint64_t *result
);

int inkling_json_number_i64(
    const InklingJsonValue *value,
    int64_t *result
);

int inkling_json_number_double(
    const InklingJsonValue *value,
    double *result
);

const char *inkling_json_number_text(
    const InklingJsonValue *value,
    size_t *length
);

const char *inkling_json_string(
    const InklingJsonValue *value,
    size_t *length
);

size_t inkling_json_array_size(const InklingJsonValue *value);

const InklingJsonValue *inkling_json_array_at(
    const InklingJsonValue *value,
    size_t index
);

size_t inkling_json_object_size(const InklingJsonValue *value);

const InklingJsonValue *inkling_json_object_get(
    const InklingJsonValue *value,
    const char *key
);

const char *inkling_json_object_key_at(
    const InklingJsonValue *value,
    size_t index,
    size_t *length
);

const InklingJsonValue *inkling_json_object_value_at(
    const InklingJsonValue *value,
    size_t index
);

#endif
