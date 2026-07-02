#pragma once

#include "common/storage.h"
#include "common/tensor.h"

#include <memory>
#include <string>
#include <vector>

struct MetalBackend {
    struct Impl;

    static void init();
    // Fresh uninitialized buffer; the storage owns it.
    static std::shared_ptr<Storage> allocate_buffer(size_t size);
    // Wrap existing memory, zero copy. ptr must be page-aligned (e.g. mmap)
    // and outlive the storage.
    static std::shared_ptr<Storage> allocate_buffer(void* ptr, size_t size);
    static Tensor dispatch(const std::string& kernel_name, std::vector<Tensor> tensors);
};
