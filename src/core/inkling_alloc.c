#include <stdlib.h>

#include "inkling_alloc.h"

int inkling_size_add(size_t a, size_t b, size_t *out)
{
    if (out == NULL || b > SIZE_MAX - a) {
        return 0;
    }
    *out = a + b;
    return 1;
}

int inkling_size_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (b != 0 && a > SIZE_MAX / b)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

int inkling_alloc_array(uint64_t count, size_t element_size, void **out)
{
    if (out == NULL) {
        return 0;
    }
    *out = NULL;
    size_t bytes = 0;
    if (element_size == 0 || count > SIZE_MAX ||
        !inkling_size_mul((size_t)count, element_size, &bytes) ||
        (uintmax_t)bytes > (uintmax_t)PTRDIFF_MAX) {
        return 0;
    }
    if (bytes == 0) {
        return 1;
    }
    *out = malloc(bytes);
    return *out != NULL;
}
