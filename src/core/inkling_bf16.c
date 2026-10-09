#include <float.h>
#include <limits.h>
#include <string.h>

#include "inkling_bf16.h"

#if CHAR_BIT != 8 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 || \
    FLT_MIN_EXP != -125 || FLT_MAX_EXP != 128
#error "BF16 widening requires IEEE-754 binary32 float"
#endif
typedef char InklingFloatMustBe32Bits[sizeof(float) == sizeof(uint32_t) ? 1 : -1];

float inkling_bf16_to_f32(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}
