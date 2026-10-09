#ifndef INKLING_NVFP4_H
#define INKLING_NVFP4_H

#include <stddef.h>
#include <stdint.h>

/* Checkpoint order is [expert, row, column], with no stored padding/swizzle. */
typedef struct {
    size_t experts, rows, columns;
    size_t packed_bytes, scale_bytes, scale2_bytes;
} InklingNvfp4Layout;

typedef struct {
    size_t packed, scale, scale2; /* Byte offsets within each separate tensor. */
} InklingNvfp4Offsets;

/* Validate original_shape, U8 packed shape, F8_E4M3 scale shape and F32 scale2
 * count. Rank must be 3 for the shape arrays; scale2 must be a rank-1 vector.
 * Only positive dimensions and a final dimension divisible by 16 are supported.
 * Return 1 on success; otherwise return 0 and clear a non-NULL output.
 */
int inkling_nvfp4_layout(const uint64_t original[3], const uint64_t packed[3],
                        const uint64_t scales[3], uint64_t scale2_count,
                        InklingNvfp4Layout *out);

/* Use a layout returned by inkling_nvfp4_layout. No payload access/allocation.
 * Offsets select 8 packed bytes, 1 scale byte, and 4 scale2 bytes for one block.
 * Return 0 for invalid coordinates/overflow and clear a non-NULL output.
 */
int inkling_nvfp4_offsets(const InklingNvfp4Layout *layout, size_t expert,
                         size_t row, size_t block, InklingNvfp4Offsets *out);

/* E2M1 codes 0..15 (others return NaN); E4M3FN accepts every byte.
 * Both preserve signed zero. E4M3FN has signed NaNs at 0x7f/0xff, no infinities.
 */
float inkling_e2m1_to_f32(uint8_t code);
float inkling_e4m3_to_f32(uint8_t code);

/* Decode exactly 16 values from at least 8 packed bytes, low nibble first.
 * Compute local_scale * scale2 in float32, then multiply each E2M1 value.
 * Require finite nonnegative local scale, finite positive scale2, and at least
 * 16 output slots. On error (including nonfinite results), leave output intact.
 * Caller supplies scale2 for the selected expert, loaded from its own shard.
 */
int inkling_nvfp4_decode_block(const uint8_t *packed, size_t packed_bytes,
                              uint8_t scale, float scale2, float *out,
                              size_t out_count);

#endif
