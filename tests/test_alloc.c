#include <stdio.h>
#include <stdlib.h>

#include "../src/core/inkling_alloc.h"

static int failures;
static int fail_allocation;
static size_t allocation_calls;
static size_t last_allocation_size;

/* Only inkling_alloc.c's test object redirects malloc here. The production
 * source has no allocator hooks; failure is deterministic without exhausting RAM.
 */
void *inkling_test_malloc(size_t bytes)
{
    allocation_calls++;
    last_allocation_size = bytes;
    return fail_allocation ? NULL : malloc(bytes);
}

static void expect(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

int main(void)
{
    size_t result = 17;
    expect(inkling_size_add(0, 0, &result) && result == 0, "zero sum");
    expect(inkling_size_add(SIZE_MAX, 0, &result) && result == SIZE_MAX, "maximum sum");
    expect(inkling_size_add(SIZE_MAX - 1, 1, &result) && result == SIZE_MAX,
           "sum at boundary");
    result = 17;
    expect(!inkling_size_add(SIZE_MAX, 1, &result) && result == 17,
           "sum overflow preserves output");
    expect(!inkling_size_add(1, SIZE_MAX, &result) && result == 17,
           "reversed sum overflow");
    expect(!inkling_size_add(1, 2, NULL), "NULL sum output");

    expect(inkling_size_mul(0, SIZE_MAX, &result) && result == 0, "zero product");
    expect(inkling_size_mul(SIZE_MAX, 0, &result) && result == 0, "reversed zero product");
    expect(inkling_size_mul(SIZE_MAX, 1, &result) && result == SIZE_MAX, "maximum product");
    expect(inkling_size_mul(SIZE_MAX / 2, 2, &result) && result == SIZE_MAX - 1,
           "product below boundary");
    result = 17;
    expect(!inkling_size_mul(SIZE_MAX / 2 + 1, 2, &result) && result == 17,
           "product overflow preserves output");
    expect(!inkling_size_mul(2, SIZE_MAX, &result) && result == 17,
           "reversed product overflow");
    expect(!inkling_size_mul(1, 2, NULL), "NULL product output");

    /* A borrowed sentinel checks output clearing without leaking an allocation. */
    int sentinel = 0;
    void *buffer = &sentinel;
    expect(inkling_alloc_array(0, sizeof(float), &buffer) && buffer == NULL,
           "empty array succeeds with NULL");
    buffer = &sentinel;
    expect(!inkling_alloc_array(1, 0, &buffer) && buffer == NULL,
           "zero element size rejected");
    buffer = &sentinel;
    expect(!inkling_alloc_array(0, 0, &buffer) && buffer == NULL,
           "empty array still requires valid element size");
    expect(!inkling_alloc_array(1, sizeof(float), NULL), "NULL allocation output");
    buffer = &sentinel;
    expect(!inkling_alloc_array(UINT64_MAX, 2, &buffer) && buffer == NULL,
           "unrepresentable allocation rejected");
    buffer = &sentinel;
    expect(!inkling_alloc_array(2, SIZE_MAX / 2 + 1, &buffer) && buffer == NULL,
           "allocation byte multiplication overflow rejected");
    buffer = &sentinel;
    expect(!inkling_alloc_array(1, SIZE_MAX, &buffer) && buffer == NULL,
           "allocation beyond pointer difference range rejected");
#if SIZE_MAX < UINT64_MAX
    buffer = &sentinel;
    expect(!inkling_alloc_array((uint64_t)SIZE_MAX + 1, 1, &buffer) && buffer == NULL,
           "count conversion cannot truncate");
#endif
    expect(allocation_calls == 0, "invalid and empty arrays never call malloc");

    fail_allocation = 1;
    buffer = &sentinel;
    expect(!inkling_alloc_array(4, sizeof(float), &buffer) && buffer == NULL,
           "malloc failure clears output");
    expect(allocation_calls == 1, "malloc failure path exercised");
    fail_allocation = 0;
    expect(inkling_alloc_array(4, sizeof(float), &buffer) && buffer != NULL,
           "successful allocation after failure");
    expect(allocation_calls == 2 && last_allocation_size == 4 * sizeof(float),
           "exact allocation size");
    if (buffer != NULL) {
        float *values = buffer;
        for (size_t i = 0; i < 4; i++) {
            values[i] = (float)i + 0.5f;
        }
        expect(values[0] == 0.5f && values[3] == 3.5f, "complete buffer is writable");
    }
    free(buffer);
    if (failures == 0) {
        puts("allocation tests passed");
    }
    return failures != 0;
}
