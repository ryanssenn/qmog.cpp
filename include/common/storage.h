#pragma once

#include <cstddef>
#include <memory>

enum class Device {
    CPU,
    Metal,
};

struct Storage {
    Device device;
    Storage(Device d) : device(d) {}

    virtual ~Storage() = default;
    virtual void* cpu_ptr() = 0;
};

struct CPUStorage : public Storage {
    void* data;
    bool owned;

    // Wrap existing memory (mmap'd weights). Non-owning.
    explicit CPUStorage(void* d) : Storage(Device::CPU), data(d), owned(false) {}

    // Allocate fresh memory (arena backing). Owning.
    explicit CPUStorage(size_t size)
        : Storage(Device::CPU), data(new char[size]), owned(true) {}

    ~CPUStorage() override {
        if (owned) delete[] static_cast<char*>(data);
    }

    void* cpu_ptr() override {
        return data;
    }
};

// Single entry point for storage creation.
// ptr != nullptr: wrap existing memory, zero copy (Metal requires page-aligned ptr).
// ptr == nullptr: allocate size fresh bytes, owned by the storage.
std::shared_ptr<Storage> make_storage(Device device, void* ptr, size_t size);
