# Tensor Storage Design

How `Tensor` went from a raw `void* data` pointer to a device-aware
`Storage` abstraction, and how the arena and the model loader build on it.

## The problem

The original `Tensor` held a `void* data` pointer. That worked while all
memory was host memory, but it cannot represent GPU memory: a Metal
buffer is an `id<MTLBuffer>` object, not a pointer, and kernel dispatch
needs the buffer object itself (`setBuffer:offset:`), not an address.
Supporting Metal meant `Tensor` needed to know *which allocation* its
data lives in, not just where the bytes are.

## Decision 1: Tensor = (storage, byte offset)

```
struct Tensor {
    ...
    std::shared_ptr<Storage> storage;
    size_t offset = 0;
};
```

A tensor never owns memory. It identifies an allocation (`storage`) and
a position within it (`offset`, in bytes). The element pointer is
`storage->cpu_ptr() + offset`.

- Views (`at`, `reshape`) share the parent's storage and adjust the
  offset. No copies, no ownership transfer.
- `shared_ptr` keeps the allocation alive as long as any tensor
  references it. Lifetime management needs no explicit code.
- The `(storage, offset)` pair is exactly what Metal kernel dispatch
  consumes: `[encoder setBuffer:buf offset:t.offset atIndex:i]`.

This is the same structure PyTorch (`c10::StorageImpl` + `DataPtr` +
byte offset) and ggml/llama.cpp (`ggml_backend_buffer` + offset)
converged on.

## Decision 2: one Storage base class, one concrete type per device

```
struct Storage {
    Device device;              // CPU or Metal
    virtual void* cpu_ptr() = 0;
};

struct CPUStorage   : Storage { void* data; bool owned; };  // storage.h
struct MetalStorage : Storage { id<MTLBuffer> data; };      // metal_backend.mm only
```

`Storage` answers one question: where do the bytes live and who frees
them. Nothing else.

- `cpu_ptr()` works for both devices because Apple Silicon has unified
  memory: `MTLResourceStorageModeShared` buffers are host-visible via
  `[data contents]`. CPU code (`f32()`, `f16()`, `copy_from`, tests)
  is device-blind.
- `MetalStorage` is defined inside `metal_backend.mm` because it holds
  an Objective-C type that cannot appear in a C++ header. The rest of
  the codebase compiles as plain C++ and only sees `Storage`.

## Decision 3: a single creation entry point

```
// storage.cpp — the only pure-C++ file that includes metal_backend.h
std::shared_ptr<Storage> make_storage(Device device, void* ptr, size_t size);
```

One rule, identical on both devices:

- `ptr == nullptr` — allocate `size` fresh bytes, **owned** by the
  storage. CPU: `new char[size]`. Metal: `newBufferWithLength:`.
- `ptr != nullptr` — **wrap** existing memory, zero copy, non-owning;
  the caller guarantees lifetime. CPU: alias the pointer. Metal:
  `newBufferWithBytesNoCopy:` (requires a page-aligned pointer, which
  mmap guarantees).

An earlier version copied on the Metal wrap path (`newBufferWithBytes:`)
because per-weight pointers were not page-aligned. That asymmetry —
same call meaning "alias" on CPU and "copy" on Metal — was removed once
the loader switched to wrapping the whole file (Decision 5). Keeping
both devices on the same semantics means callers never need to reason
about the device.

Routing creation through one function also keeps `metal_backend.h` out
of `tensor.h`/`arena.h`, avoiding a circular include (`metal_backend.h`
itself uses `Tensor`).

## Decision 4: the arena owns one Storage and hands out offsets

```
struct Arena {
    std::shared_ptr<Storage> storage;   // created once: make_storage(device, nullptr, size)
    size_t capacity;
    size_t offset = 0;

    Arena(size_t size, Device device = Device::CPU);
    size_t allocate(size_t size);       // returns a byte offset, not a pointer
};
```

The arena is a bump allocator for activations. Its job is integer
arithmetic — align (64 bytes), bump, bounds-check — and that logic is
device-independent. The only device-specific part is acquiring the
backing block, which happens once in the constructor via `make_storage`.
There is no `CPUArena`/`MetalArena` split; the polymorphism lives in
`Storage`, where it already exists.

`allocate()` returns a byte offset instead of a pointer because a
pointer cannot express "which buffer" for Metal. `Tensor::alloc`
assembles the pair:

```
Tensor::alloc(arena, dtype, dims)
    -> make_tensor(arena.storage, arena.allocate(bytes), dtype, shape)
```

Consequences:

- All activation tensors (`hidden_state`, `k_cache`, ...) share the
  arena's single storage at distinct offsets. On Metal that is one
  `MTLBuffer` for all activations — one allocation at startup, no
  per-tensor Metal objects.
- `Tensor::alloc` needs no device parameter. The arena's device,
  chosen once by `InferenceState`, decides.
- 64-byte alignment already exceeds Metal's offset-alignment
  requirement for `setBuffer:offset:`.

Trade-off accepted: tensors within an arena have no isolation (an
overrun in one region silently corrupts a neighbor, and address
sanitizers cannot see it). This is the standard trade for inference
activations, which have identical lifetimes and fixed sizes; the golden
tests are the mitigation.

## Decision 5: weights are views into one mmap-backed Storage

The loader mmaps the model file once and wraps the whole mapping in a
single storage:

```
storage = make_storage(device, base, file_size);        // in ModelLoad::load
Tensor t = Tensor::view(storage, payload_base + offset, dtype, shape);
```

Previously each weight called `from_ptr(base + offset)`, baking the file
offset into a raw pointer and creating a separate `CPUStorage` per
weight. That blocked zero-copy Metal loading: `newBufferWithBytesNoCopy`
requires page alignment, which individual weight addresses do not have —
only the mmap base does.

Keeping the offset as data instead of pointer arithmetic gives:

- **Zero-copy Metal weights.** One `newBufferWithBytesNoCopy` over the
  page-aligned mapping; the `MTLBuffer` *is* the file mapping (same
  physical pages). No 2x memory during load, no upload step.
- **Honest aliasing.** All weights share one storage, so "same storage"
  means "same memory" — the invariant PyTorch documents for its
  storages.
- **Symmetry with the arena.** Activations: one storage, offsets
  assigned by the bump allocator. Weights: one storage, offsets
  assigned by the file format.

The mapping must outlive the buffer (nothing is copied), which holds
because the mapping lives for the process lifetime.

## Decision 6: ownership is a property of the storage, not the tensor

`CPUStorage` carries an `owned` flag: memory it allocated is freed in
its destructor; wrapped memory (the mmap) is not. `MetalStorage` needs
no flag — ARC releases the `MTLBuffer` when the storage dies, and for
wrapped (NoCopy) buffers the underlying pages belong to the mapping.
This is the minimal version of the deleter-function pattern PyTorch
uses; a flag suffices while there are exactly two cleanup behaviors.

## Initialization

`MetalBackend::init()` (device + command queue) is called once at each
entry point (`src/main.cpp`, `test/qwen/main.cpp`) and is idempotent.
Storage creation assumes the device exists; nothing initializes lazily.

## Memory map after load

```
mmap'd model file  ->  1 Storage  ->  ~300 weight tensors, offsets from the file header
arena (one block)  ->  1 Storage  ->  16 activation tensors, offsets from the bump allocator
```

Every tensor in the system is a view: storage identifies the
allocation, offset locates the data, and on Metal that pair maps
directly onto `setBuffer:offset:` when kernels dispatch.
