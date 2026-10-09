#include <stdio.h>
#include <string.h>

#include "../src/core/inkling_bf16.h"

int main(void)
{
    /* Includes both zeros, every subnormal/normal, infinities, and all quiet and
     * signaling NaN payloads of either sign. Inspect bits without evaluating NaNs.
     */
    for (uint32_t pattern = 0; pattern <= UINT16_MAX; pattern++) {
        float value = inkling_bf16_to_f32((uint16_t)pattern);
        uint32_t bits = 0;
        memcpy(&bits, &value, sizeof(bits));
        if ((bits >> 16) != pattern || (bits & UINT32_C(0xffff)) != 0) {
            fprintf(stderr, "FAIL: BF16 pattern 0x%04x widened to 0x%08x\n",
                    (unsigned int)pattern, (unsigned int)bits);
            return 1;
        }
    }
    if (inkling_bf16_to_f32(UINT16_C(0x3f80)) != 1.0f ||
        inkling_bf16_to_f32(UINT16_C(0xc000)) != -2.0f ||
        inkling_bf16_to_f32(UINT16_C(0x0001)) != 0x1p-133f ||
        inkling_bf16_to_f32(UINT16_C(0x0080)) != 0x1p-126f) {
        fputs("FAIL: BF16 known numerical values\n", stderr);
        return 1;
    }
    puts("all 65536 BF16 bit patterns passed");
    return 0;
}
