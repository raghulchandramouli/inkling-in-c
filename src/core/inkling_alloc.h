#ifndef INKLING_ALLOC_H
#define INKLING_ALLOC_H

#include <stddef.h>
#include <stdint.h>

/* Return 1 on success, 0 on overflow or NULL output. Failure leaves *out unchanged. */
int inkling_size_add(size_t a, size_t b, size_t *out);
int inkling_size_mul(size_t a, size_t b, size_t *out);

/* Allocate count elements of element_size bytes, with malloc's alignment.
 * Return 1 on success, 0 on invalid size/output or allocation failure.
 * count == 0 succeeds with *out == NULL; element_size == 0 is always invalid.
 * Reject counts beyond SIZE_MAX and byte lengths beyond SIZE_MAX or PTRDIFF_MAX.
 * A non-NULL out is cleared on entry, including on failure. It must not hold an
 * unreleased allocation. On success the caller owns uninitialized storage and
 * releases it with free(). Use a void * temporary, not a cast from T ** to void **.
 */
int inkling_alloc_array(uint64_t count, size_t element_size, void **out);

#endif
