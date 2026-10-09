#ifndef INKLING_BF16_H
#define INKLING_BF16_H

#include <stdint.h>

/* Bit-exact widening on IEEE-754 binary32 platforms: preserve sign, exponent,
 * and all payload bits, including signaling NaNs. No floating-point arithmetic.
 */
float inkling_bf16_to_f32(uint16_t value);

#endif
