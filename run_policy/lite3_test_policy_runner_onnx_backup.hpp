/**
 * @file lite3_test_policy_runner_onnx.hpp
 * @brief ONNX runner for two-leg stand policy (117-D obs × 40 history = 4797 total)
 */

#ifndef LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
#define LITE3_TEST_POLICY_RUNNER_ONNX_HPP_

#include "policy_runner_base.hpp"
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>

#include <deque>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <fstream>   // for debug dumps
#include <cmath>

using namespace types;

class Lite3TestPolicyRunnerONNX : public PolicyRunnerBase {
private:
    // === ONNX session ===
    std::string       model_path_;
    Ort::Env          env_;
    Ort::SessionOptions session_options_;
    Ort::Session      session_;
    Ort::MemoryInfo   memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // === Policy & robot constants ===
    static constexpr int   obs_dim_         = 117;   // per frame
    static constexpr int   obs_hist_num_    = 40;
    static constexpr int   total_input_dim_ = obs_dim_ * (1 + obs_hist_num_);
    static constexpr int   act_dim_         = 12;
    static constexpr int   motor_num        = 12;
    static constexpr float kActionScaleTrain = 0.25f;  // PPO training action_scale

    // === Normalization (two_leg_stand) ===
    static constexpr float kAngVelScale   = 1.0f;
    static constexpr float kOrientScale   = 1.0f;   // roll/pitch/yaw
    static constexpr float kLinVelScale   = 1.0f;   // commands
    static constexpr float kDofPosScale   = 1.0f;
    static constexpr float kDofVelScale   = 0.1f;   // IMPORTANT

    // === Observation buffers ===
    VecXf               current_obs_;                 // 117-D
    std::deque<VecXf>   history_obs_;                 // 40 frames

    // Per-feature short histories inside each 117-D frame
    std::deque<VecXf>   pos_hist_;    // up to 3×12
    std::deque<VecXf>   vel_hist_;    // up to 2×12
    std::deque<VecXf>   tgt_hist_;    // up to 2×12 (RAW policy actions)

    // === Control & mapping ===
    VecXf joint_pos_rl = VecXf(act_dim_);
    VecXf joint_vel_rl = VecXf(act_dim_);
    VecXf last_action_hist = VecXf::Zero(act_dim_);
    VecXf tmp_action   = VecXf::Zero(act_dim_);
    VecXf action       = VecXf::Zero(act_dim_);

    VecXf dof_pos_default_policy, dof_pos_default_robot;
    VecXf kp_, kd_;

    // kept for compatibility; not used in obs construction (training uses 1.0)
    Vec3f max_cmd_vel_{0.8, 0.8, 0.8};

    std::vector<int>   robot2policy_idx, policy2robot_idx;
    std::vector<float> action_scale_robot{
        0.125f, 0.25f, 0.25f,
        0.125f, 0.25f, 0.25f,
        0.125f, 0.25f, 0.25f,
        0.125f, 0.25f, 0.25f};

    RobotAction ra;

    bool history_seeded_ = false;

    // === Helper ===
    std::vector<int> generate_permutation(
        const std::vector<std::string>& from,
        const std::vector<std::string>& to,
        int default_index = 0)
    {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < (int)from.size(); ++i)
            idx_map[from[i]] = i;

        std::vector<int> perm;
        perm.reserve(to.size());
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            perm.push_back(it != idx_map.end() ? it->second : default_index);
        }
        return perm;
    }

public:
    Lite3TestPolicyRunnerONNX(std::string policy_name)
        : PolicyRunnerBase(policy_name),
          env_(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy"),
          session_options_(),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          session_(nullptr)
    {
        model_path_ = GetAbsPath() + "/../policy/ppo/policy.onnx";
        std::cout << "[ONNX INIT] Loading model: " << model_path_ << std::endl;
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
        std::cout << "[ONNX INIT] Model loaded successfully.\n";

        input_names_  = {"obs"};
        output_names_ = {"action"};

        dof_pos_default_policy.resize(act_dim_);
        dof_pos_default_policy <<
            0.0,  -0.8, 1.6,
            0.0,  -0.8, 1.6,
            0.0,  -0.8, 1.6,
            0.0,  -0.8, 1.6;

        dof_pos_default_robot = dof_pos_default_policy;

        kp_ = 30.f * VecXf::Ones(12);
        kd_ =  1.f * VecXf::Ones(12);

        // identity mapping (adjust if your robot order differs)
        const std::vector<std::string> order{
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};
        robot2policy_idx = generate_permutation(order, order);
        policy2robot_idx = robot2policy_idx;

        ra.goal_joint_pos = VecXf::Zero(act_dim_);
        ra.goal_joint_vel = VecXf::Zero(act_dim_);
        ra.tau_ff         = VecXf::Zero(act_dim_);
        ra.kp             = kp_;
        ra.kd             = kd_;

        decimation_ = 12; // keep as in original repo
    }

    ~Lite3TestPolicyRunnerONNX() {}

    void DisplayPolicyInfo() override {
        std::cout << "ONNX policy: " << policy_name_ << "\n";
        std::cout << "path: " << model_path_ << "\n";
        std::cout << "obs_dim: " << obs_dim_ << ", action_dim: " << act_dim_ << "\n";
    }

    void OnEnter() override {
        run_cnt_ = 0;
        current_obs_.setZero(obs_dim_);

        // === Reset all histories to zeros (critical for correct adaptation) ===
        history_obs_.clear();
        pos_hist_.clear();
        vel_hist_.clear();
        tgt_hist_.clear();

        // Fill with zeros (match training init)
        VecXf zero_obs = VecXf::Zero(obs_dim_);
        for (int i = 0; i < obs_hist_num_; ++i)
            history_obs_.push_back(zero_obs);

        joint_pos_rl.setZero(act_dim_);
        joint_vel_rl.setZero(act_dim_);
        last_action_hist.setZero(act_dim_);
        history_seeded_ = false;

        std::cout << "[ONNX ENTER] History cleared. PolicyRunner entered: "
                  << policy_name_ << std::endl;
    }

    RobotAction GetRobotAction(const RobotBasicState& ro) override {
        static bool first = true;
        if (first) {
        first = false;
        std::cout << "[RAW] cmd_norm: " << ro.cmd_vel_normlized.transpose() << "\n";
        std::cout << "[RAW] base_rpy: " << ro.base_rpy.transpose() << "\n";
        std::cout << "[RAW] base_omega: " << ro.base_omega.transpose() << "\n";
        std::cout << "[RAW] joint_pos[0..11]: " << ro.joint_pos.head<12>().transpose() << "\n";
        std::cout << "[RAW] joint_vel[0..11]: " << ro.joint_vel.head<12>().transpose() << "\n";
        std::cout << "[RAW] R (row-major): "
                    << ro.base_rot_mat.row(0) << " | "
                    << ro.base_rot_mat.row(1) << " | "
                    << ro.base_rot_mat.row(2) << "\n";
        }

        // === 1) Build current 117-D frame (match training order & scaling) ===

        // (a) commands — use normalized commands as-is (no extra max_cmd_vel factor)
        Vec3f cmd = ro.cmd_vel_normlized * kLinVelScale;

        // (b) orientation (roll, pitch, yaw) as in training
        Vec3f base_rpy = ro.base_rpy * kOrientScale;

        // (c) base angular velocity (body frame)
        Vec3f base_ang_vel_world = ro.base_omega;
        Vec3f base_ang_vel = ro.base_rot_mat.transpose() * base_ang_vel_world;
        base_ang_vel *= kAngVelScale;

        // (d) joints (policy joint order), centered by nominal and scaled
        for (int i = 0; i < act_dim_; ++i) {
            const int p = robot2policy_idx[i];

            joint_pos_rl(i) = ro.joint_pos(p) * kDofPosScale;
            joint_vel_rl(i) =  ro.joint_vel(p) * kDofVelScale; // 0.1 scale
        }

        // === 2) Short “within-frame” histories (oldest→newest) ===
        if (!history_seeded_ && pos_hist_.empty()) {
            for (int i = 0; i < 3; ++i) pos_hist_.push_back(joint_pos_rl);
        } else {
            pos_hist_.push_back(joint_pos_rl);
            if ((int)pos_hist_.size() > 3) pos_hist_.pop_front();
        }

        if (!history_seeded_ && vel_hist_.empty()) {
            for (int i = 0; i < 2; ++i) vel_hist_.push_back(joint_vel_rl);
        } else {
            vel_hist_.push_back(joint_vel_rl);
            if ((int)vel_hist_.size() > 2) vel_hist_.pop_front();
        }

        // === Target history seed ===
        VecXf action_hist_frame = VecXf::Zero(act_dim_);
        if (!history_seeded_) {
            action_hist_frame = joint_pos_rl - dof_pos_default_policy;
        } else {
            action_hist_frame = last_action_hist;
        }
        if (!history_seeded_ && tgt_hist_.empty()) {
            for (int i = 0; i < 2; ++i) tgt_hist_.push_back(action_hist_frame);
        } else {
            tgt_hist_.push_back(action_hist_frame);
            if ((int)tgt_hist_.size() > 2) tgt_hist_.pop_front();
        }
        VecXf pos_hist_flat(36); pos_hist_flat.setZero();
        VecXf vel_hist_flat(24); vel_hist_flat.setZero();
        VecXf tgt_hist_flat(24); tgt_hist_flat.setZero();

        int idx = 0;
        for (const auto& v : pos_hist_) { pos_hist_flat.segment(idx, 12) = v; idx += 12; }
        idx = 0;
        for (const auto& v : vel_hist_) { vel_hist_flat.segment(idx, 12) = v; idx += 12; }
        idx = 0;
        for (const auto& v : tgt_hist_) { tgt_hist_flat.segment(idx, 12) = v; idx += 12; }

        // Concatenate 117:
        // [cmd(3), proj_g(3), base_ang_vel(3), dof_pos(12), dof_vel(12), pos_hist(36), vel_hist(24), tgt_hist(24)]
        current_obs_.resize(obs_dim_);
        current_obs_ << cmd, base_rpy, base_ang_vel,
                        joint_pos_rl, joint_vel_rl,
                        pos_hist_flat, vel_hist_flat, tgt_hist_flat;

        // === 3) Concatenate current frame + 40-step history ===
        VecXf input_flat(total_input_dim_);
        input_flat.segment(0, obs_dim_) = current_obs_;

        int offset = obs_dim_;
        for (const auto& frame : history_obs_) {
            input_flat.segment(offset, obs_dim_) = frame;
            offset += obs_dim_;
        }

        // === 4) ONNX inference ===
        std::array<int64_t, 2> input_shape{1, total_input_dim_};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_flat.data(), total_input_dim_,
            input_shape.data(), input_shape.size());

        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    input_names_.data(), &input_tensor, 1,
                                    output_names_.data(), 1);

        float* act_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> act(act_data, act_dim_);
        action = act;                                // RAW policy action (policy joint order)
        last_action_hist = action * kActionScaleTrain;
        history_seeded_ = true;

        // Slide history after using current frame
        history_obs_.push_back(current_obs_);
        if ((int)history_obs_.size() > obs_hist_num_) history_obs_.pop_front();

        // === 5) Map to robot order and PD target ===
        for (int i = 0; i < act_dim_; ++i) {
            tmp_action(i) = action(policy2robot_idx[i]);   // reorder to robot joint order
            tmp_action(i) *= action_scale_robot[i];        // per-joint PD amplitude
        }
        tmp_action += dof_pos_default_robot;

        ra.goal_joint_pos = tmp_action;
        ra.goal_joint_vel = VecXf::Zero(act_dim_);
        ra.tau_ff         = VecXf::Zero(act_dim_);
        ra.kp             = kp_;
        ra.kd             = kd_;

        ++run_cnt_;

        // --- Debug export (first 10 steps) ---
        if (run_cnt_ <= 10) {
            std::ofstream f("debug_cpp_step" + std::to_string(run_cnt_-1) + ".txt");
            f << "obs_flat";
            for (int i = 0; i < total_input_dim_; ++i) f << " " << input_flat[i];
            f << "\naction";
            for (int i = 0; i < act_dim_; ++i) f << " " << action[i];
            f.close();
        }

        return ra;
    }
};

#endif // LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
