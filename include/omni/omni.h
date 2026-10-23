#ifndef OMNI_TOKENIZER_H
#define OMNI_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(OMNI_BUILD)
#define OMNI_API __declspec(dllexport)
#elif defined(_WIN32)
#define OMNI_API __declspec(dllimport)
#else
#define OMNI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct omni_model omni_model;
typedef struct omni_context omni_context;
typedef void (*omni_trace_callback)(void *user, const char *stage, const float *values,
                                    size_t rows, size_t columns);

enum omni_status {
    OMNI_OK = 0, OMNI_INVALID_ARGUMENT = 1, OMNI_IO_ERROR = 2,
    OMNI_INVALID_MODEL = 3, OMNI_UNSUPPORTED = 4, OMNI_OUT_OF_MEMORY = 5,
    OMNI_CANCELLED = 6, OMNI_INTERNAL_ERROR = 7
};
enum omni_backend { OMNI_CPU = 0, OMNI_CUDA = 1 };
enum omni_kind { OMNI_AUDIO = 1, OMNI_IMAGE = 2 };

typedef struct omni_model_info {
    uint32_t struct_size;
    uint32_t kind;
    uint32_t layers;
    uint32_t width;
    uint32_t heads;
    uint32_t code_dim;
    uint32_t vocabulary_size;
    uint32_t backend;
    uint64_t weight_bytes;
} omni_model_info;

/* Error text is thread local and valid until the next call on that thread. */
OMNI_API const char *omni_last_error(void);
OMNI_API const char *omni_version(void);
OMNI_API int omni_has_cuda(void);
/* Paths are UTF-8. A context retains its model after the model handle is freed. */
OMNI_API int omni_model_load(const char *path, int backend, int device, omni_model **out);
OMNI_API void omni_model_free(omni_model *model);
OMNI_API int omni_model_get_info(const omni_model *model, omni_model_info *out);
OMNI_API int omni_context_create(omni_model *model, omni_context **out);
OMNI_API void omni_context_free(omni_context *context);
/* One inference per context at a time. Separate contexts share immutable weights. */
OMNI_API void omni_context_cancel(omni_context *context);
OMNI_API void omni_context_reset_cancel(omni_context *context);
/* Optional synchronous diagnostics. Values are borrowed for the callback duration.
 * The callback must not throw or invoke inference on this context. */
OMNI_API int omni_context_set_trace(omni_context *context, omni_trace_callback callback, void *user);
/* mel is contiguous [mel_channels,frames], frames must be a multiple of 8.
 * valid_frames is ceil(valid_samples / 160), and codes_needed=ceil(valid_frames/8).
 * Canonical preprocessing pads each waveform to a multiple of 1280 samples.
 * Output is caller-owned; count is set on success, or to required capacity. */
OMNI_API int omni_encode_audio(omni_context *context, const float *mel,
                              uint32_t frames, uint32_t valid_frames,
                              uint16_t *codes, size_t capacity, size_t *count);
/* pixels is contiguous RGB CHW, preprocessed to [-1,1]. H/W are patch multiples.
 * The code grid has (height/patch_size,width/patch_size), in row-major order. */
OMNI_API int omni_encode_image(omni_context *context, const float *pixels,
                              uint32_t height, uint32_t width,
                              uint16_t *codes, size_t capacity, size_t *count);
/* Copy the last pre-VQ features as float32 [code_count,code_dim].
 * Image features have been L2 normalized. Used for numerical validation. */
OMNI_API int omni_context_copy_features(omni_context *context, float *output,
                                       size_t capacity, size_t *count);
/* Peak live activation/workspace allocation within this context, excluding weights. */
OMNI_API uint64_t omni_context_peak_workspace(const omni_context *context);

#ifdef __cplusplus
}
#endif
#endif
