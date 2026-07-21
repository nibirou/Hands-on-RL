#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available:  " << (torch::cuda::is_available() ? "YES" : "NO") << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
        
        // 最硬核的测试：直接在 GPU (CUDA) 上创建一个随机张量
        // 如果这一步不报错，说明 CUDA 环境 100% 配置成功！
        auto options = torch::TensorOptions().device(torch::kCUDA);
        auto x = torch::randn({3, 3}, options);
        
        std::cout << "Successfully created GPU Tensor:\n" << x << std::endl;
    } else {
        std::cout << "CUDA is NOT available. Falling back to CPU." << std::endl;
        auto x = torch::randn({3, 3});
        std::cout << "CPU Tensor:\n" << x << std::endl;
    }
    
    return 0;
}
// 编译命令
// g++ -std=c++17 test_torch.cpp \
//     -I /workspace/env/libtorch/include \
//     -I /workspace/env/libtorch/include/torch/csrc/api/include \
//     -L /workspace/env/libtorch/lib \
//     -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
//     -Wl,-rpath,/workspace/env/libtorch/lib \
//     -o test_torch.out