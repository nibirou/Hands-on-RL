#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <memory>   // 【新增】用于 std::unique_ptr
#include <tuple>    // 【新增】用于 std::get

#include <torch/torch.h>

// ==========================================
// 1. 自定义 CartPole 环境
// ==========================================
class CartPole {
private:
    std::array<float, 4> state;
    int step_count;
    std::mt19937 gen;

    const float gravity = 9.8;
    const float mass_cart = 1.0;
    const float mass_pole = 0.1;
    const float total_mass = mass_cart + mass_pole;
    const float length = 0.5; 
    const float polemass_length = mass_pole * length;
    const float force_mag = 10.0;
    const float tau = 0.02;  
    const float theta_threshold_radians = 12.0 * 2.0 * M_PI / 360.0;
    const float x_threshold = 2.4;
    const int max_steps = 500;

public:
    CartPole(int seed = 0) : gen(seed) { reset(); }

    std::vector<float> reset() {
        std::uniform_real_distribution<float> dis(-0.05, 0.05);
        state = {dis(gen), dis(gen), dis(gen), dis(gen)};
        step_count = 0;
        return std::vector<float>(state.begin(), state.end());
    }

    std::tuple<std::vector<float>, float, bool, bool> step(int action) {
        float x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
        float force = (action == 1) ? force_mag : -force_mag;
        float costheta = std::cos(theta);
        float sintheta = std::sin(theta);
        
        float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        float thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0f/3.0f - mass_pole * costheta * costheta / total_mass));
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        
        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;
        
        state = {x, x_dot, theta, theta_dot};
        step_count++;
        
        bool terminated = (x < -x_threshold || x > x_threshold || 
                           theta < -theta_threshold_radians || theta > theta_threshold_radians);
        bool truncated = (step_count >= max_steps);
        
        return {std::vector<float>(state.begin(), state.end()), 1.0f, terminated, truncated};
    }
};

// ==========================================
// 2. 经验回放池
// ==========================================
struct Transition {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class ReplayBuffer {
private:
    std::deque<Transition> buffer;
    size_t capacity;
    std::mt19937 sampler_gen;

public:
    ReplayBuffer(size_t cap, int seed = 0) : capacity(cap), sampler_gen(seed) {}

    void add(const Transition& t) {
        if (buffer.size() == capacity) buffer.pop_front();
        buffer.push_back(t);
    }

    std::vector<Transition> sample(size_t batch_size) {
        std::vector<size_t> indices(buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), sampler_gen);
        
        std::vector<Transition> batch;
        batch.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer[indices[i]]);
        }
        return batch;
    }

    size_t size() const { return buffer.size(); }
};

// ==========================================
// 3. Q 网络
// ==========================================
class QNetImpl : public torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
public:
    QNetImpl(int state_dim, int hidden_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, action_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);
    }
};
TORCH_MODULE(QNet);

// ==========================================
// 4. DQN 智能体 (彻底修复 state_dict 报错)
// ==========================================
class DQNAgent {
    private:
        QNet q_net = nullptr;
        QNet target_q_net = nullptr;
        std::unique_ptr<torch::optim::Adam> optimizer; 
        
        float gamma;
        float epsilon;
        int target_update;
        int count;
        torch::Device device;
        int action_dim;
    
        // 【新增辅助函数】：直接通过 parameters() 和 copy_() 复制权重
        void copy_weights() {
            auto q_params = q_net->parameters();
            auto target_params = target_q_net->parameters();
            // 确保两个网络参数数量一致
            for (size_t i = 0; i < q_params.size(); ++i) {
                target_params[i].copy_(q_params[i]);
            }
        }
    
    public:
        DQNAgent(int state_dim, int hidden_dim, int action_dim, float lr, float gamma, 
                 float epsilon, int target_update, torch::Device device)
            : gamma(gamma), epsilon(epsilon), target_update(target_update), 
              count(0), device(device), action_dim(action_dim) {
            
            q_net = QNet(state_dim, hidden_dim, action_dim);
            target_q_net = QNet(state_dim, hidden_dim, action_dim);
            
            q_net->to(device);
            target_q_net->to(device);
            
            // 【修改点 1】：使用 copy_weights() 替代 load_state_dict
            copy_weights();
            target_q_net->eval();
            
            optimizer = std::make_unique<torch::optim::Adam>(
                q_net->parameters(), 
                torch::optim::AdamOptions(lr)
            );
        }
    
        int take_action(const std::vector<float>& state) {
            if (torch::rand({1}).item<float>() < epsilon) {
                return torch::randint(0, action_dim, {1}).item<int>();
            } else {
                torch::NoGradGuard no_grad;
                auto state_tensor = torch::tensor(state, torch::kFloat32).unsqueeze(0).to(device);
                auto q_values = q_net->forward(state_tensor);
                return q_values.argmax(1).item<int>();
            }
        }
    
        void update(const std::vector<Transition>& batch) {
            int b_size = batch.size();
            int s_dim = batch[0].state.size();
            
            std::vector<float> states_f, rewards_f, next_states_f, dones_f;
            std::vector<int64_t> actions_i;
            
            for(const auto& t : batch) {
                states_f.insert(states_f.end(), t.state.begin(), t.state.end());
                actions_i.push_back(t.action);
                rewards_f.push_back(t.reward);
                next_states_f.insert(next_states_f.end(), t.next_state.begin(), t.next_state.end());
                dones_f.push_back(t.done ? 1.0f : 0.0f);
            }
            
            auto states = torch::tensor(states_f, torch::kFloat32).reshape({b_size, s_dim}).to(device);
            auto actions = torch::tensor(actions_i, torch::kInt64).reshape({b_size, 1}).to(device);
            auto rewards = torch::tensor(rewards_f, torch::kFloat32).reshape({b_size, 1}).to(device);
            auto next_states = torch::tensor(next_states_f, torch::kFloat32).reshape({b_size, s_dim}).to(device);
            auto dones = torch::tensor(dones_f, torch::kFloat32).reshape({b_size, 1}).to(device);
    
            auto q_values = q_net->forward(states).gather(1, actions);
            
            auto max_out = target_q_net->forward(next_states).max(1);
            auto max_next_q = std::get<0>(max_out).reshape({b_size, 1});
            
            auto q_targets = rewards + gamma * max_next_q * (1.0 - dones);
    
            auto loss = torch::mse_loss(q_values, q_targets);
            
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
    
            count++;
            if (count % target_update == 0) {
                // 【修改点 2】：使用 copy_weights() 替代 load_state_dict
                copy_weights();
            }
        }
    };

// ==========================================
// 5. 辅助函数与主程序
// ==========================================
std::vector<float> moving_average(const std::vector<float>& data, int window) {
    std::vector<float> ma;
    for (int i = 0; i < data.size(); ++i) {
        int start = std::max(0, i - window + 1);
        float sum = 0;
        for (int j = start; j <= i; ++j) sum += data[j];
        ma.push_back(sum / (i - start + 1));
    }
    return ma;
}

void print_progress(int current, int total, float avg_return) {
    int bar_width = 40;
    float progress = (float)current / total;
    int pos = bar_width * progress;
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% "
              << "Ep: " << current << "/" << total 
              << " | Avg Return: " << std::fixed << std::setprecision(2) << avg_return << std::flush;
}

int main() {
    float lr = 2e-3;
    int num_episodes = 500;
    int hidden_dim = 128;
    float gamma = 0.98;
    float epsilon = 0.01;
    int target_update = 10;
    int buffer_size = 10000;
    int minimal_size = 500;
    int batch_size = 64;

    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    std::srand(0);
    torch::manual_seed(0);

    CartPole env(0);
    ReplayBuffer replay_buffer(buffer_size, 0);
    
    int state_dim = 4;
    int action_dim = 2;
    DQNAgent agent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device);

    std::vector<float> return_list;
    return_list.reserve(num_episodes);

    std::cout << "Starting DQN Training on CartPole-v1..." << std::endl;
    for (int i_episode = 0; i_episode < num_episodes; ++i_episode) {
        float episode_return = 0;
        auto state = env.reset();
        bool done = false;

        while (!done) {
            int action = agent.take_action(state);
            auto [next_state, reward, terminated, truncated] = env.step(action);
            done = terminated || truncated;

            replay_buffer.add({state, action, reward, next_state, done});
            state = next_state;
            episode_return += reward;

            if (replay_buffer.size() > minimal_size) {
                auto batch = replay_buffer.sample(batch_size);
                agent.update(batch);
            }
        }
        return_list.push_back(episode_return);

        if ((i_episode + 1) % 10 == 0 || i_episode == 0) {
            float recent_avg = 0;
            int window = std::min(10, (int)return_list.size());
            for(int i=0; i<window; ++i) recent_avg += return_list[return_list.size() - 1 - i];
            recent_avg /= window;
            print_progress(i_episode + 1, num_episodes, recent_avg);
        }
    }
    std::cout << "\nTraining finished!" << std::endl;

    std::ofstream csv_file("dqn_returns.csv");
    csv_file << "episode,return,moving_avg\n";
    auto mv_return = moving_average(return_list, 9);
    for (size_t i = 0; i < return_list.size(); ++i) {
        csv_file << i + 1 << "," << return_list[i] << "," << mv_return[i] << "\n";
    }
    csv_file.close();
    std::cout << "Results saved to dqn_returns.csv" << std::endl;

    return 0;
}