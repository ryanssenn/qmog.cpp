cmake_minimum_required(VERSION 3.20)
project(mog.cpp CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

if(APPLE)
    set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${LIBOMP_ROOT}/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LIBOMP_ROOT}/lib -lomp")
else()
    find_package(OpenMP REQUIRED)
endif()

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE TEST_MISTRAL CONFIGURE_DEPENDS "test/mistral/*.cpp")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_executable(mog-cli src/main.cpp ${SRC})
add_executable(test_exec ${SRC} ${TEST_MISTRAL})

if(NOT APPLE)
    target_link_libraries(mog-cli PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test_exec PRIVATE OpenMP::OpenMP_CXX)
endif()
