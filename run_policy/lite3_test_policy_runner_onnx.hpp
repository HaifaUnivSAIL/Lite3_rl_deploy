/**
 * @file lite3_test_policy_runner_onnx.hpp
 * @brief ONNX policy runner with history stacking and runtime IO discovery
 *        (updated to support 117-dim obs + 40-frame history = 4797 total)
 */

#ifndef LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
#define LITE3_TEST_POLICY_RUNNER_ONNX_HPP_

#include "policy_runner_base.hpp"
#include <onnxruntime_cxx_api.h>

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <iostream>
#include <array>

using namespace types;
using VecXf = Eigen::VectorXf;
using Vec3f = Eigen::Vector3f;

class Lite3TestPolicyRunnerONNX : public PolicyRunnerBase {
private:
    // ----- model / ORT -----
    std::string model_path_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions alloc_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    // Names discovered from the ONNX (don’t hard-code "obs"/"actions")
    std::string input_name_;
    std::string output_name_;

    // Input dims from ONNX
    int64_t input_dim_total_{-1};  // e.g., 4797

    // ----- observation/action layout -----
    // IMPORTANT: Set to the *current* single-frame obs size your model expects
    // Your trained model uses 117 (and history length derived below).
    static constexpr int kObsDim = 117;
    static constexpr int kActDim = 12;

    // We keep the original 45 signals you already compute, and zero-fill the rest (72)
    static constexpr int kObsDimRunnerLegacy = 45;
    static_assert(kObsDim >= kObsDimRunnerLegacy, "kObsDim must be >= legacy runner obs size");

    int obs_dim_{kObsDim};   // single-frame obs count (117)
    int act_dim_{kActDim};   // action dim (12)
    int hist_n_{0};          // history length derived from ONNX input: (total - obs_dim) / obs_dim

    // Rolling history of past single-frame observations (each length = obs_dim_)
    std::deque<VecXf> history_;

    // ----- working buffers -----
    VecXf current_obs_;      // [obs_dim_]
    VecXf last_action_;      // [act_dim_]
    VecXf joint_pos_rl_;     // [act_dim_], policy order
    VecXf joint_vel_rl_;     // [act_dim_], policy order

    VecXf tmp_action_;       // [act_dim_]
    VecXf action_;           // [act_dim_]
    Vec3f gravity_direction_ = Vec3f(0.f, 0.f, -1.f);

    VecXf dof_pos_default_robot_, dof_pos_default_policy_;
    VecXf kp_, kd_;
    Vec3f max_cmd_vel_;
    float omega_scale_   = 0.25f;
    float dof_vel_scale_ = 0.05f;

    std::vector<int> robot2policy_idx_, policy2robot_idx_;
    std::vector<std::string> robot_order_{
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"
    };
    std::vector<std::string> policy_order_ = robot_order_;

    std::vector<float> action_scale_robot_{
        0.125f, 0.25f, 0.25f,
        0.125f, 0.25f, 0.25f,
        0.125f, 0.25f, 0.25f,
        0.125f, 0.25f, 0.25f
    };

    RobotAction ra_;

    // ----- helpers -----
    static std::vector<int> generate_permutation(
        const std::vector<std::string>& from,
        const std::vector<std::string>& to,
        int default_index = 0)
    {
        std::unordered_map<std::string,int> idx;
        for (int i = 0; i < (int)from.size(); ++i) idx[from[i]] = i;
        std::vector<int> perm; perm.reserve(to.size());
        for (const auto& name : to) {
            auto it = idx.find(name);
            perm.push_back(it == idx.end() ? default_index : it->second);
        }
        return perm;
    }

    void discover_io_and_histlen_() {
        // input/output names
        input_name_  = session_.GetInputNameAllocated(0, alloc_).get();
        output_name_ = session_.GetOutputNameAllocated(0, alloc_).get();

        // input dims
        auto ti   = session_.GetInputTypeInfo(0);
        auto tsv  = ti.GetTensorTypeAndShapeInfo();
        auto dims = tsv.GetShape(); // expect {1, N}
        if (dims.size() != 2 || dims[0] != 1) {
            std::cerr << "[ONNX] Unexpected input rank; got ";
            for (auto d : dims) std::cerr << d << " ";
            std::cerr << std::endl;
        }
        input_dim_total_ = dims[1];

        // derive history length: total = obs_dim + obs_dim*hist_n
        const int64_t remainder = input_dim_total_ - obs_dim_;
        if (remainder >= 0 && remainder % obs_dim_ == 0) {
            hist_n_ = static_cast<int>(remainder / obs_dim_);
        } else {
            // Fallback: assume no history if model shape doesn’t align
            hist_n_ = 0;
        }

        std::cout << "[ONNX IO] input='" << input_name_
                  << "' total=" << input_dim_total_
                  << " obs_dim=" << obs_dim_
                  << " hist_n=" << hist_n_ << std::endl;

        // (Optional) verify output dims once
        auto to = session_.GetOutputTypeInfo(0);
        auto tov = to.GetTensorTypeAndShapeInfo();
        auto odims = tov.GetShape(); // e.g., {1, 12}
        std::cout << "[ONNX IO] output='" << output_name_
                  << "' shape=[" << (odims.size() > 0 ? odims[0] : -1)
                  << "," << (odims.size() > 1 ? odims[1] : -1) << "]" << std::endl;
    }

    // Build a 117-dim single-frame obs from current robot state
    // We fill the first 45 with your existing features, and zero-fill the remaining 72 for now.
    // Later we’ll replace the zero-fill with the exact features from training.
    void build_current_obs_(const RobotBasicState& ro) {
        // Precompute basic terms (same as your original code)
        Vec3f base_omega = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction_;
        Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(max_cmd_vel_);

        // joints to policy order
        for (int i = 0; i < act_dim_; ++i) {
            joint_pos_rl_(i) = ro.joint_pos(robot2policy_idx_[i]);
            joint_vel_rl_(i) = ro.joint_vel(robot2policy_idx_[i]) * dof_vel_scale_;
        }
        joint_pos_rl_ -= dof_pos_default_policy_;

        // ---- assemble the first 45 entries exactly as before ----
        // legacy layout: [base_omega(3), projected_gravity(3), cmd_vel(3),
        //                joint_pos_rl(12), joint_vel_rl(12), last_action(12)] = 45
        int off = 0;
        current_obs_.segment(off, 3) = base_omega;              off += 3;
        current_obs_.segment(off, 3) = projected_gravity;       off += 3;
        current_obs_.segment(off, 3) = cmd_vel;                 off += 3;
        current_obs_.segment(off, 12) = joint_pos_rl_;          off += 12;
        current_obs_.segment(off, 12) = joint_vel_rl_;          off += 12;
        current_obs_.segment(off, 12) = last_action_;           off += 12;

        // ---- zero-fill the remaining (117 - 45) entries for now ----
        if (off < obs_dim_) {
            current_obs_.segment(off, obs_dim_ - off).setZero();
        }
    }

    // Maintain the history ring and build the flat [1, total] tensor data
    VecXf build_flat_input_() {
        // push current frame to history
        if (hist_n_ > 0) {
            if ((int)history_.size() == hist_n_) history_.pop_front();
            history_.push_back(current_obs_);
        }

        VecXf flat(input_dim_total_);
        // current first
        flat.segment(0, obs_dim_) = current_obs_;

        int off = obs_dim_;
        if (hist_n_ > 0) {
            // NOTE: order must match training. We use oldest->newest here.
            for (const auto& h : history_) {
                flat.segment(off, obs_dim_) = h;
                off += obs_dim_;
            }
        }
        // pad zeros during warmup
        if (off < input_dim_total_) {
            flat.segment(off, input_dim_total_ - off).setZero();
        }
        return flat;
    }

public:
    explicit Lite3TestPolicyRunnerONNX(std::string policy_name)
    : PolicyRunnerBase(policy_name)
    , env_(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy")
    , session_options_()
    , session_(nullptr)
    {
        // Model path (same as your original)
        model_path_ = GetAbsPath() + "/../policy/ppo/policy.onnx";
        std::cout << "[ONNX INIT] Loading model: " << model_path_ << std::endl;

        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
        std::cout << "[ONNX INIT] Model loaded successfully.\n";

        // Discover IO + compute history length from model
        discover_io_and_histlen_();

        // Defaults
        dof_pos_default_policy_.setZero(kActDim);
        dof_pos_default_policy_ <<  0.0000, -0.8000, 1.6000,
                                    0.0000, -0.8000, 1.6000,
                                    0.0000, -0.8000, 1.6000,
                                    0.0000, -0.8000, 1.6000;
        dof_pos_default_robot_ = dof_pos_default_policy_;

        kp_ = 30.f * VecXf::Ones(kActDim);
        kd_ =  1.f * VecXf::Ones(kActDim);
        max_cmd_vel_ << 0.8f, 0.8f, 0.8f;

        joint_pos_rl_ = VecXf(kActDim);
        joint_vel_rl_ = VecXf(kActDim);
        tmp_action_   = VecXf(kActDim);
        action_       = VecXf(kActDim);
        last_action_  = VecXf::Zero(kActDim);

        ra_.goal_joint_pos = VecXf::Zero(kActDim);
        ra_.goal_joint_vel = VecXf::Zero(kActDim);
        ra_.tau_ff         = VecXf::Zero(kActDim);
        ra_.kp = kp_;
        ra_.kd = kd_;

        robot2policy_idx_ = generate_permutation(robot_order_, policy_order_);
        policy2robot_idx_ = generate_permutation(policy_order_, robot_order_);
        for (int i = 0; i < kActDim; ++i) {
            std::cout << "robot2policy_idx[" << i << "]: " << robot2policy_idx_[i] << std::endl;
            std::cout << "policy2robot_idx[" << i << "]: " << policy2robot_idx_[i] << std::endl;
        }

        // lightweight smoke test: feed zeros (correct total length)
        {
            VecXf dummy = VecXf::Zero(input_dim_total_);
            std::array<int64_t,2> shape{1, input_dim_total_};
            Ort::Value in = Ort::Value::CreateTensor<float>(
                memory_info_, dummy.data(), input_dim_total_, shape.data(), shape.size());
            const char* in_names[]  = { input_name_.c_str() };
            const char* out_names[] = { output_name_.c_str() };
            auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
            (void)outs;
            std::cout << policy_name_ << " ONNX policy network test success\n";
        }

        decimation_ = 12;
    }

    ~Lite3TestPolicyRunnerONNX() {}

    void DisplayPolicyInfo() override {
        std::cout << "ONNX policy: " << policy_name_ << "\n";
        std::cout << "path: " << model_path_ << "\n";
        std::cout << "obs_dim(single): " << obs_dim_
                  << ", hist_n: " << hist_n_
                  << ", action_dim: " << act_dim_ << "\n";
    }

    void OnEnter() override {
        run_cnt_ = 0;
        history_.clear();
        current_obs_.setZero(obs_dim_);
        last_action_.setZero(act_dim_);
        std::cout << "[ONNX ENTER] PolicyRunner entered: " << policy_name_ << std::endl;
    }

    RobotAction GetRobotAction(const RobotBasicState& ro) override {
        // 1) Build single-frame obs (117)
        current_obs_.resize(obs_dim_);
        build_current_obs_(ro);

        // 2) Build flat stacked input [1 × input_dim_total_]
        VecXf flat = build_flat_input_();

        // 3) Run ONNX
        std::array<int64_t,2> input_shape{1, input_dim_total_};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, flat.data(), input_dim_total_, input_shape.data(), input_shape.size());

        const char* in_names[]  = { input_name_.c_str() };
        const char* out_names[] = { output_name_.c_str() };

        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

        float* action_data = output_tensors[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::MatrixXf> act_map(action_data, act_dim_, 1);
        action_ = VecXf(act_map);
        last_action_ = action_;

        // 4) Map to robot order + scaling + PD references
        for (int i = 0; i < act_dim_; ++i) {
            tmp_action_(i) = action_(policy2robot_idx_[i]) * action_scale_robot_[i];
        }
        tmp_action_ += dof_pos_default_robot_;

        ra_.goal_joint_pos = tmp_action_;
        ra_.goal_joint_vel = VecXf::Zero(act_dim_);
        ra_.tau_ff         = VecXf::Zero(act_dim_);
        ra_.kp             = kp_;
        ra_.kd             = kd_;

        ++run_cnt_;
        return ra_;
    }
};

#endif  // LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
