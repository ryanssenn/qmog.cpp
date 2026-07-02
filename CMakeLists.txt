cmake_minimum_required(VERSION 3.20)
project(qmog.cpp LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

set(QMOG_COMMON_SRC
    src/common/shape.cpp
    src/common/storage.cpp
    src/common/tensor.cpp
)

set(QMOG_LOADER_SRC
    src/loader/model_load.cpp
)

set(QMOG_TOKENIZER_SRC
    src/tokenizer/bpe.cpp
    src/tokenizer/qwen_pre_tokenize.cpp
    src/tokenizer/qwen_tokenizer.cpp
)

set(QMOG_MODEL_SRC
    src/model/inference_state.cpp
    src/model/model.cpp
)

set(QMOG_BACKEND_SRC
    src/backend/metal/kernels.cpp
    src/backend/metal/metal_backend.mm
)
set_source_files_properties(src/backend/metal/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

set(QMOG_DECODE_SRC
    src/decode/sampling.cpp
)

set(QMOG_EVAL_SRC
    src/eval/perplexity.cpp
)

set(TEST_QWEN_SRC
    test/qwen/main.cpp
    test/qwen/setup/context.cpp
    test/qwen/test_model_load.cpp
    test/qwen/test_pre_tokenize.cpp
    test/qwen/test_tensor.cpp
    test/qwen/test_tokenizer.cpp
)

add_executable(qmog-cli
    src/main.cpp
    ${QMOG_COMMON_SRC}
    ${QMOG_LOADER_SRC}
    ${QMOG_TOKENIZER_SRC}
    ${QMOG_MODEL_SRC}
    ${QMOG_BACKEND_SRC}
    ${QMOG_DECODE_SRC}
    ${QMOG_EVAL_SRC}
)

add_executable(test_exec
    ${QMOG_COMMON_SRC}
    ${QMOG_LOADER_SRC}
    ${QMOG_TOKENIZER_SRC}
    ${QMOG_MODEL_SRC}
    ${QMOG_BACKEND_SRC}
    ${TEST_QWEN_SRC}
)

target_include_directories(qmog-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(test_exec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/qwen
)

target_link_libraries(qmog-cli PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_link_libraries(test_exec PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
