#ifndef INKLING_MATH_H
#define INKLING_MATH_H

#include "inkling_nvfp4.h"

/* Scalar float32 reference operations. Return 1 on success, 0 on invalid
 * arguments/nonfinite arithmetic. All dimensions (including k/tokens) must be
 * positive. Counts are available elements, except packed_bytes/scale_bytes.
 * Size/NULL errors leave outputs/state intact; numerical errors may leave a
 * partial result. Discard outputs and reset convolution state after failure.
 * Writable buffers must not overlap other buffers, except exact out == x for RMSNorm, activations,
 * softmax and convolution. Partial overlap is unsupported. Callers own storage.
 * Inputs must be finite except softmax's -INFINITY mask and top-k's infinities.
 */
int inkling_bf16_matvec(const uint16_t *weights, size_t weight_count,
                        size_t rows, size_t columns, const float *x,
                        size_t x_count, float *out, size_t out_count);

/* Full contiguous [expert,row,column] buffers, one selected expert. Uses B2's
 * validated, unpadded layout; global scales are native float32 per expert.
 * Only one 16-value block is decoded at a time; no allocation/unpacked expert.
 */
int inkling_nvfp4_matvec(const InklingNvfp4Layout *layout, size_t expert,
                         const uint8_t *packed, size_t packed_bytes,
                         const uint8_t *scales, size_t scale_bytes,
                         const float *scale2, size_t scale2_count,
                         const float *x, size_t x_count,
                         float *out, size_t out_count);

/* One vector/head per call. Float32 x/weights, double sum of squares, positive
 * finite epsilon; multiply normalized x by the learned weights (no +1 offset).
 */
int inkling_rmsnorm(const float *x, size_t n, const float *weights,
                     size_t weight_count, float epsilon,
                     float *out, size_t out_count);
int inkling_sigmoid(const float *x, size_t n, float *out, size_t out_count);
int inkling_silu(const float *x, size_t n, float *out, size_t out_count);

/* Subtract maximum; -INFINITY masks become zero. Reject NaN, +INFINITY and
 * all-masked vectors. Accumulate the exponential sum in double precision.
 */
int inkling_softmax(const float *x, size_t n, float *out, size_t out_count);

/* Indices sorted by descending value, ties by ascending index (including
 * signed-zero and infinity ties). NaN rejected. 1 <= k <= n.
 */
int inkling_topk(const float *x, size_t n, size_t k,
                  size_t *indices, size_t indices_count);

/* Causal depthwise cross-correlation, kernel [channels,4], oldest tap first.
 * x/out are [tokens,channels]; state is [channels,3], oldest sample first.
 * Initialize/reset state to zeros. Each output is conv(x) + x, as in the pinned
 * InklingShortConvolution for K, V, attention and MLP. No bias or activation.
 * The attention/MLP outer layer residual is added separately by the caller.
 * Passing tokens=1 repeatedly is equivalent to a prefill/chunk call.
 */
int inkling_conv4(const float *x, size_t x_count, size_t tokens, size_t channels,
                   const float *weights, size_t weight_count,
                   float *state, size_t state_count, float *out, size_t out_count);

#endif
