#ifndef INKLING_MODEL_H
#define INKLING_MODEL_H
#include "inkling.h"
typedef struct InklingModel InklingModel;
typedef struct InklingContext InklingContext;
/* Constructors clear *out and publish an owned object only on success. */
int inkling_model_open(const char *directory, InklingModel **out);
/* Free all contexts before closing their model. */
void inkling_model_close(InklingModel *model);
size_t inkling_model_logits_width(const InklingModel *model);
int inkling_context_create(const InklingModel *model, size_t max_tokens, InklingContext **out);
void inkling_context_reset(InklingContext *context);
void inkling_context_free(InklingContext *context);
/* Every forward starts a fresh sequence. Discard logits after any failure. */
int inkling_forward(InklingContext *context, const uint32_t *ids, size_t tokens,
                    float *logits, size_t logits_count);
#endif
