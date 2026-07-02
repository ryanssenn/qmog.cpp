#include "loader/binary_reader.h"
#include "loader/model_load.h"
#include "common/shape.h"
#include <iostream>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

void* ModelLoad::map_file(int fd, size_t& size){
    struct stat st;
    if (fstat(fd, &st) < 0){
        std::cerr << "Model binary get size failed" << std::endl;
        std::exit(1);
    }

    size = st.st_size;

    void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (p == MAP_FAILED){
        std::cerr << "Model mmap failed" << std::endl;
        std::exit(1);
    }

    return p;
}

void ModelLoad::load_config(BinaryReader& reader){
    uint32_t count = reader.read_u32();

    for (uint32_t i = 0; i < count; i++){
        std::string key = reader.read_string();
        uint8_t type = reader.read_u8();

        if (key == "hidden_size" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.hidden_size = reader.read_u32();
        } else if (key == "intermediate_size" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.intermediate_size = reader.read_u32();
        } else if (key == "n_layers" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.n_layers = reader.read_u32();
        } else if (key == "n_heads" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.n_heads = reader.read_u32();
        } else if (key == "n_kv_heads" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.n_kv_heads = reader.read_u32();
        } else if (key == "head_dim" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.head_dim = reader.read_u32();
        } else if (key == "vocab_size" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.vocab_size = reader.read_u32();
        } else if (key == "sliding_window" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.sliding_window = reader.read_u32();
        } else if (key == "max_position_embeddings" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.max_position_embeddings = reader.read_u32();
        } else if (key == "rope_theta" && type == static_cast<uint8_t>(model_format::ConfigValueType::FLOAT32)){
            config.rope_theta = reader.read_f32();
        } else if (key == "norm_eps" && type == static_cast<uint8_t>(model_format::ConfigValueType::FLOAT32)){
            config.norm_eps = reader.read_f32();
        } else if (key == "quant" && type == static_cast<uint8_t>(model_format::ConfigValueType::STRING)){
            config.quant = reader.read_string();
        } else if (key == "tie_word_embeddings" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.tie_word_embeddings = reader.read_u32() != 0;
        } else if (key == "bos_token_id" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.bos_token_id = reader.read_u32();
        } else if (key == "eos_token_id" && type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
            config.eos_token_id = reader.read_u32();
        } else {
            if (type == static_cast<uint8_t>(model_format::ConfigValueType::STRING)){
                reader.read_string();
            } else if (type == static_cast<uint8_t>(model_format::ConfigValueType::UINT32)){
                reader.read_u32();
            } else if (type == static_cast<uint8_t>(model_format::ConfigValueType::FLOAT32)){
                reader.read_f32();
            }
        }
    }

    if (config.head_dim == 0) {
        config.head_dim = config.hidden_size / config.n_heads;
    }
}

void ModelLoad::load_tensor(std::unordered_map<std::string, Tensor>& m, const std::string& key, uint8_t dtype, const std::vector<size_t>& shape, uint64_t offset){
    DType dt = dtype_from_file(dtype);
    Shape s = Shape::from_dims(shape);
    Tensor t = Tensor::view(storage, offset, dt, s);
    m.insert({key, t});
}

void ModelLoad::load_weights(size_t payload_base, BinaryReader& reader){
    layer_weights.resize(config.n_layers);

    uint32_t count = reader.read_u32();
    const std::string model_prefix = "model.layers.";

    for (uint32_t i = 0; i < count; i++){
        std::string key = reader.read_string();
        uint8_t dtype = reader.read_u8();
        uint8_t ndim = reader.read_u8();

        std::vector<size_t> shape;
        shape.reserve(ndim);
        for (int d = 0; d < 4; d++){
            uint32_t dim = reader.read_u32();
            if (d < ndim){
                shape.push_back(dim);
            }
        }

        uint64_t offset = reader.read_u64();
        // Per-tensor scale fields remain in the format; unused without INT8.
        reader.read_u64();
        reader.read_u32();

        if (key.compare(0, model_prefix.size(), model_prefix) == 0){
            std::string rest = key.substr(model_prefix.size());
            int layer = std::stoi(rest.substr(0, rest.find(".")));
            load_tensor(layer_weights[layer], rest.substr(rest.find(".") + 1), dtype, shape, payload_base + offset);
            continue;
        }

        load_tensor(global_weights, key, dtype, shape, payload_base + offset);
    }
}

void ModelLoad::load(const std::string& path, Device device){
    int fd = open(path.c_str(), O_RDONLY);

    if (fd < 0){
        std::cerr << "Model binary open failed" << std::endl;
        std::exit(1);
    }

    size_t file_size = 0;
    void* mapped = map_file(fd, file_size);
    char* base = static_cast<char*>(mapped);

    if (file_size < model_format::FILE_PREFIX_SIZE){
        std::cerr << "unsupported format: file too small" << std::endl;
        std::exit(1);
    }

    if (std::memcmp(base, model_format::MAGIC, 4) != 0){
        std::cerr << "unsupported format: expected MOG magic" << std::endl;
        std::exit(1);
    }

    uint32_t version;
    std::memcpy(&version, base + 4, sizeof(version));
    if (version != model_format::FORMAT_VERSION){
        std::cerr << "unknown format version: " << version << std::endl;
        std::exit(1);
    }

    uint64_t header_size;
    std::memcpy(&header_size, base + 8, sizeof(header_size));

    size_t header_start = model_format::FILE_PREFIX_SIZE;
    size_t header_end = header_start + header_size;
    if (header_end > file_size){
        std::cerr << "truncated header" << std::endl;
        std::exit(1);
    }

    BinaryReader reader(base + header_start, header_size);
    config.architecture = reader.read_string();
    load_config(reader);
    tokenizer.load(reader);

    size_t payload_base = (header_end + 63) & ~size_t(63);
    if (payload_base > file_size){
        std::cerr << "truncated payload" << std::endl;
        std::exit(1);
    }

    storage = make_storage(device, base, file_size);
    load_weights(payload_base, reader);

    close(fd);
}

Tensor& ModelLoad::get_tensor(int layer, const std::string& name){
    if (layer == -1){
        for (auto& e : global_weights){
            if (name == e.first){
                return e.second;
            }
        }

        assert(false && "tensor not found");
        static Tensor empty;
        return empty;
    }

    for (auto& e : layer_weights[layer]){
        if (name == e.first){
            return e.second;
        }
    }

    assert(false && "tensor not found");
    static Tensor empty;
    return empty;
}
