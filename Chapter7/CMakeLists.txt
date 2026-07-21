# 编译DQN.cpp
cmake_minimum_required(VERSION 3.18)
project(dqn_cpp)

# 强制使用 C++17 标准 (代码中使用了结构化绑定)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 【核心修改点 1】：将 CMAKE_PREFIX_PATH 指向你解压 libtorch 的绝对路径
set(CMAKE_PREFIX_PATH "/workspace/env/libtorch")

# 查找 PyTorch 包
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 生成可执行文件
add_executable(dqn_cpp DQN.cpp)
target_link_libraries(dqn_cpp "${TORCH_LIBRARIES}")

# 【核心修改点 2】：设置运行时库路径 (RPATH)，防止运行时找不到 .so 文件
set_target_properties(dqn_cpp PROPERTIES
    BUILD_RPATH "/workspace/env/libtorch/lib"
    INSTALL_RPATH "/workspace/env/libtorch/lib"
)

# 编译test_torch.cpp
# cmake_minimum_required(VERSION 3.18)
# project(torch_test)

# # 强制使用 C++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 指向你的 LibTorch 路径
# set(CMAKE_PREFIX_PATH "/workspace/env/libtorch")

# # 自动查找并配置所有 Torch 依赖 (包含处理 --no-as-needed 的魔法)
# find_package(Torch REQUIRED)

# # 生成可执行文件
# add_executable(test_torch test_torch.cpp)

# # 链接库
# target_link_libraries(test_torch "${TORCH_LIBRARIES}")

# # 确保运行时能找到库
# set_target_properties(test_torch PROPERTIES
#     BUILD_RPATH "/workspace/env/libtorch/lib"
# )