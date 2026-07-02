#include "setup/context.h"

#include "backend/metal/metal_backend.h"
#include "common/arena.h"
#include "common/tensor.h"

#include <iostream>

#include <sys/mman.h>
#include <unistd.h>

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) {
        std::cout << msg << std::endl;
    }
    return cond;
}

// Fill with i, read back and verify.
bool roundtrip(Tensor& t) {
    float* p = t.f32();
    for (size_t i = 0; i < t.numel; i++) {
        p[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < t.numel; i++) {
        if (p[i] != static_cast<float>(i)) {
            std::cout << "roundtrip mismatch at " << i << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

// from_ptr wraps existing memory without copying: writes through the tensor
// are visible in the source array.
int test_tensor_cpu_from_ptr() {
    float src[6] = {0, 1, 2, 3, 4, 5};
    Tensor t = Tensor::from_ptr(src, DType::F32, Shape::from_dims({2, 3}));

    if (!expect(t.storage->device == Device::CPU, "device should be CPU")) return 1;
    if (!expect(t.numel == 6 && t.ndim == 2, "shape should be 2x3")) return 1;
    if (!expect(t.f32() == src, "f32() should alias the source")) return 1;

    t.f32()[4] = 42.f;
    if (!expect(src[4] == 42.f, "write through tensor should hit source")) return 1;

    return 0;
}

// Arena-allocated tensors share the arena's single storage at distinct,
// aligned offsets, and hold readable/writable memory.
int test_tensor_cpu_alloc() {
    Arena arena(1 << 16);
    Tensor a = Tensor::alloc(arena, DType::F32, {8});
    Tensor b = Tensor::alloc(arena, DType::F32, {4, 4});

    if (!expect(a.storage == arena.storage, "a should share arena storage")) return 1;
    if (!expect(a.storage == b.storage, "a and b should share storage")) return 1;
    if (!expect(a.offset % Arena::kAlignment == 0, "a offset should be aligned")) return 1;
    if (!expect(b.offset >= a.offset + a.byte_size(), "b should not overlap a")) return 1;
    if (!roundtrip(a) || !roundtrip(b)) return 1;

    return 0;
}

// at() and reshape() are views: same storage, adjusted offset, shared bytes.
int test_tensor_views() {
    Arena arena(1 << 16);
    Tensor t = Tensor::alloc(arena, DType::F32, {3, 4});
    if (!roundtrip(t)) return 1;

    Tensor row = t.at({1});
    if (!expect(row.storage == t.storage, "view should share storage")) return 1;
    if (!expect(row.offset == t.offset + 4 * sizeof(float), "row offset should skip one row")) return 1;
    if (!expect(row.numel == 4 && row.ndim == 1, "row should be 1x4")) return 1;
    if (!expect(row.f32()[0] == 4.f, "row[0] should be t[1][0]")) return 1;

    row.f32()[2] = 99.f;
    if (!expect(t.f32()[6] == 99.f, "write through view should hit parent")) return 1;

    Tensor flat = t.reshape({12});
    if (!expect(flat.storage == t.storage && flat.offset == t.offset, "reshape should alias")) return 1;
    if (!expect(flat.f32()[6] == 99.f, "reshape should see the same bytes")) return 1;

    return 0;
}

// A Metal arena is one MTLBuffer; tensors view it at offsets and its memory
// is CPU-visible through cpu_ptr (unified memory).
int test_tensor_metal_alloc() {
    Arena arena(1 << 20, Device::Metal);
    Tensor a = Tensor::alloc(arena, DType::F32, {8});
    Tensor b = Tensor::alloc(arena, DType::F32, {8});

    if (!expect(a.storage->device == Device::Metal, "device should be Metal")) return 1;
    if (!expect(a.storage == arena.storage && b.storage == arena.storage,
                "tensors should share the arena MTLBuffer")) return 1;
    if (!expect(b.offset >= a.offset + a.byte_size(), "b should not overlap a")) return 1;
    if (!roundtrip(a) || !roundtrip(b)) return 1;

    return 0;
}

// from_ptr on Metal wraps page-aligned memory zero-copy: the MTLBuffer
// aliases the source pages, so both sides see each other's writes.
int test_tensor_metal_from_ptr() {
    size_t page = static_cast<size_t>(getpagesize());
    void* mem = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (!expect(mem != MAP_FAILED, "mmap failed")) return 1;

    float* src = static_cast<float*>(mem);
    for (size_t i = 0; i < 4; i++) {
        src[i] = static_cast<float>(i);
    }

    {
        Tensor t = Tensor::from_ptr(src, DType::F32, Shape::from_dims({4}), Device::Metal);

        if (!expect(t.storage->device == Device::Metal, "device should be Metal")) return 1;
        if (!expect(t.f32() == src, "Metal tensor should alias the source pages")) return 1;

        src[3] = -1.f;
        if (!expect(t.f32()[3] == -1.f, "source write should be visible through tensor")) return 1;

        t.f32()[0] = 42.f;
        if (!expect(src[0] == 42.f, "tensor write should be visible in source")) return 1;
    }

    munmap(mem, page);
    return 0;
}

// Loading the model on Metal wraps the mmap'd file in one MTLBuffer:
// all weights share that storage and hold the same bytes as a CPU load.
int test_model_load_metal() {
    ModelLoad metal_model;
    metal_model.load(model_path(), Device::Metal);

    if (!expect(metal_model.storage->device == Device::Metal, "weights should be Metal")) return 1;

    Tensor& a = metal_model.get_tensor(-1, "model.norm.weight");
    Tensor& b = metal_model.get_tensor(0, "input_layernorm.weight");
    if (!expect(a.storage == b.storage, "all weights should share the file buffer")) return 1;

    Tensor& cpu = get_model()->get_tensor(-1, "model.norm.weight");
    if (!expect(a.numel == cpu.numel, "shapes should match CPU load")) return 1;
    for (size_t i = 0; i < a.numel; i++) {
        if (a.f16()[i] != cpu.f16()[i]) {
            std::cout << "weight mismatch at " << i << std::endl;
            return 1;
        }
    }

    return 0;
}

static RegisterTest tensor_cpu_from_ptr("tensor cpu from_ptr", "any", &test_tensor_cpu_from_ptr);
static RegisterTest tensor_cpu_alloc("tensor cpu alloc", "any", &test_tensor_cpu_alloc);
static RegisterTest tensor_views("tensor views", "any", &test_tensor_views);
static RegisterTest tensor_metal_alloc("tensor metal alloc", "any", &test_tensor_metal_alloc);
static RegisterTest tensor_metal_from_ptr("tensor metal from_ptr", "any", &test_tensor_metal_from_ptr);
static RegisterTest model_load_metal("model load metal", "f16", &test_model_load_metal);
