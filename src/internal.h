#pragma once
#include "omni/omni.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace omni {
struct Error : std::runtime_error {
    int code;
    Error(int c, const std::string &s) : std::runtime_error(s), code(c) {}
};
inline void require(bool condition, const std::string &message, int code = OMNI_INVALID_ARGUMENT) {
    if (!condition)
        throw Error(code, message);
}
struct Memory {
    std::atomic<uint64_t> live{0}, peak{0};
    void add(uint64_t n) {
        auto v = live.fetch_add(n) + n;
        auto p = peak.load();
        while (v > p && !peak.compare_exchange_weak(p, v)) {
        }
    }
};
struct Value {
    std::shared_ptr<void> storage;
    size_t rows = 0, cols = 0;
    bool bf16 = false;
    void *data() const { return storage.get(); }
    size_t size() const { return rows * cols; }
};
struct Tensor {
    Value value;
    std::vector<uint64_t> shape;
};
struct Config {
    uint32_t kind = 0, layers = 0, width = 0, heads = 0, ffn = 0, input_channels = 0,
             patch_size = 0, pool_size = 0, block_size = 0, pos_height = 0, pos_width = 0,
             code_dim = 0, vocab_size = 0;
};
class Ops {
  public:
    std::shared_ptr<Memory> memory = std::make_shared<Memory>();
    virtual ~Ops() = default;
    virtual void enable_reuse() {}
    virtual Value upload(const float *, size_t, size_t, bool) = 0;
    virtual Value raw_upload(const void *, size_t, size_t, bool) = 0;
    virtual std::vector<float> download(const Value &) = 0;
    virtual Value linear(const Value &, const Value &, const Value *) = 0;
    virtual Value norm(const Value &, const Value &, const Value &, float) = 0;
    virtual void gelu(Value &) = 0;
    virtual void add(Value &, const Value &) = 0;
    virtual void bias(Value &, const Value &) = 0;
    virtual Value conv_gather(const Value &, size_t) = 0;
    virtual Value attention(const Value &, const Value *, const Value *, size_t, size_t, size_t,
                            bool) = 0;
    virtual Value pool(const Value &, size_t) = 0;
    virtual Value interpolate(const Value &, size_t, size_t, size_t) = 0;
    virtual Value normalize(const Value &) = 0;
    virtual Value squares(const Value &) = 0;
    virtual std::vector<uint16_t> quantize(const Value &, const Value &, const Value &,
                                           std::atomic<bool> &) = 0;
    virtual void synchronize() = 0;
};
std::unique_ptr<Ops> make_cpu();
#ifdef OMNI_CUDA_ENABLED
std::unique_ptr<Ops> make_cuda(int device);
#endif
struct Model {
    Config cfg;
    int backend = 0, device = 0;
    uint64_t weight_bytes = 0;
    std::unordered_map<std::string, Tensor> tensors;
    Value codebook_norms;
    const Value &w(const std::string &name) const {
        auto it = tensors.find(name);
        require(it != tensors.end(), "Missing tensor: " + name, OMNI_INVALID_MODEL);
        return it->second.value;
    }
    const Value *optional(const std::string &name) const {
        auto i = tensors.find(name);
        return i == tensors.end() ? nullptr : &i->second.value;
    }
};
struct Context {
    std::shared_ptr<Model> model;
    std::unique_ptr<Ops> ops;
    std::atomic<bool> cancelled{false};
    std::atomic_flag running = ATOMIC_FLAG_INIT;
    Value features;
    omni_trace_callback trace = nullptr;
    void *trace_user = nullptr;
    void check() const {
        if (cancelled.load())
            throw Error(OMNI_CANCELLED, "Inference cancelled");
    }
};
std::shared_ptr<Model> load(const char *path, int backend, int device);
std::unique_ptr<Ops> make_ops(int backend, int device);
std::vector<uint16_t> audio(Context &, const float *, uint32_t, uint32_t);
std::vector<uint16_t> image(Context &, const float *, uint32_t, uint32_t);
inline float round_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } v{f};
    if ((v.u & 0x7fffffffU) > 0x7f800000U)
        v.u |= 0x00400000U;
    else
        v.u += 0x7fffU + ((v.u >> 16) & 1U);
    v.u &= 0xffff0000U;
    return v.f;
}
} // namespace omni
