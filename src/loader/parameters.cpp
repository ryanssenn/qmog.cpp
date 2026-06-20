#include "../../include/parameters.h"
#include "../../include/model_format.h"
#include "../../include/binary_reader.h"
#include <iostream>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

void* Parameters::map_file(int fd, size_t& size){
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

void Parameters::load_config(BinaryReader& reader){
    uint32_t count = reader.read_u32();

    for (uint32_t i = 0; i < count; i++){
        std::string key = reader.read_string();
        uint8_t type = reader.read_u8();

        if (key == "hidden_size" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.hidden_size = reader.read_u32();
        } else if (key == "intermediate_size" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.intermediate_size = reader.read_u32();
        } else if (key == "n_layers" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.n_layers = reader.read_u32();
        } else if (key == "n_heads" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.n_heads = reader.read_u32();
        } else if (key == "n_kv_heads" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.n_kv_heads = reader.read_u32();
        } else if (key == "vocab_size" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.vocab_size = reader.read_u32();
        } else if (key == "sliding_window" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.sliding_window = reader.read_u32();
        } else if (key == "max_position_embeddings" && type == static_cast<uint8_t>(model_format::KVType::UINT32)){
            config.max_position_embeddings = reader.read_u32();
        } else if (key == "rope_theta" && type == static_cast<uint8_t>(model_format::KVType::FLOAT32)){
            config.rope_theta = reader.read_f32();
        } else if (key == "norm_eps" && type == static_cast<uint8_t>(model_format::KVType::FLOAT32)){
            config.norm_eps = reader.read_f32();
        } else if (key == "quant" && type == static_cast<uint8_t>(model_format::KVType::STRING)){
            config.quant = reader.read_string();
        } else {
            if (type == static_cast<uint8_t>(model_format::KVType::STRING)){
                reader.read_string();
            } else if (type == static_cast<uint8_t>(model_format::KVType::UINT32)){
                reader.read_u32();
            } else if (type == static_cast<uint8_t>(model_format::KVType::FLOAT32)){
                reader.read_f32();
            }
        }
    }

    config.head_dim = config.hidden_size / config.n_heads;
}

void Parameters::load_tensor(std::unordered_map<std::string, std::variant<Tensor<float>, Tensor<int8_t>>>& m, char* p, const std::string& key, uint8_t dtype, const std::vector<size_t>& shape, uint64_t offset, uint64_t scale_offset, uint32_t scale_size){
    if (dtype == static_cast<uint8_t>(model_format::DType::F32)){
        Tensor<float> t = Tensor(reinterpret_cast<float*>(p + offset), shape);
        m.insert({key, t});
    }
    else if (dtype == static_cast<uint8_t>(model_format::DType::INT8)){
        float* scale_start = reinterpret_cast<float*>(p + scale_offset);
        Tensor<int8_t> t = Tensor(reinterpret_cast<int8_t*>(p + offset), std::vector<float>(scale_start, scale_start + scale_size), shape);
        m.insert({key, t});
    }
}

void Parameters::load_weights(char* p, BinaryReader& reader){
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
        uint64_t scale_offset = reader.read_u64();
        uint32_t scale_size = reader.read_u32();

        if (key.compare(0, model_prefix.size(), model_prefix) == 0){
            std::string rest = key.substr(model_prefix.size());
            int layer = std::stoi(rest.substr(0, rest.find(".")));
            load_tensor(layer_weights[layer], p, rest.substr(rest.find(".") + 1), dtype, shape, offset, scale_offset, scale_size);
            continue;
        }

        load_tensor(global_weights, p, key, dtype, shape, offset, scale_offset, scale_size);
    }
}

void Parameters::load_parameters(const std::string& path){
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
    reader.read_string(); // architecture (e.g. "mistral")
    load_config(reader);
    tokenizer.load(reader);

    size_t payload_base = (header_end + 63) & ~size_t(63);
    if (payload_base > file_size){
        std::cerr << "truncated payload" << std::endl;
        std::exit(1);
    }

    load_weights(base + payload_base, reader);

    close(fd);
}

template<typename T>
Tensor<T> Parameters::get_tensor(int layer, const std::string& name){
    if (layer == -1){
        for (auto& e : global_weights){
            if (name == e.first){
                return std::get<Tensor<T>>(e.second);
            }
        }

        assert(false && "tensor not found");
        return {};
    }

    for (auto& e : layer_weights[layer]){
        if (name == e.first){
            return std::get<Tensor<T>>(e.second);
        }
    }

    assert(false && "tensor not found");
    return {};
}

template Tensor<float> Parameters::get_tensor<float>(int, const std::string&);
template Tensor<int8_t> Parameters::get_tensor<int8_t>(int, const std::string&);

