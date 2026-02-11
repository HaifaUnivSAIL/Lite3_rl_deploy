/**
 * @file lite3_test_policy_runner_onnx.hpp
 * @brief ONNX policy runner for Lite3 two-leg stand (117-D obs × 40 history).
 *
 * Observation layout mirrors training exactly:
 *   obs = [cmd(3), rpy(3), body_omega(3),
 *          dof_pos(12), dof_vel*0.1(12),
 *          pos_hist(3×12), vel_hist(2×12), target_hist(2×12)]
 * The flattened ONNX input is [current_obs, history_frame0, ..., history_frame39]
 * with history frames managed like rsl_rl's HistoryWrapper (oldest first).
 */

#ifndef LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
#define LITE3_TEST_POLICY_RUNNER_ONNX_HPP_

#include "policy_runner_base.hpp"

#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace types;

class Lite3TestPolicyRunnerONNX : public PolicyRunnerBase {
public:
    explicit Lite3TestPolicyRunnerONNX(std::string policy_name)
        : PolicyRunnerBase(std::move(policy_name)),
          env_(ORT_LOGGING_LEVEL_WARNING, "Lite3Policy"),
          session_options_(),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        InitSession();
        InitRobotConstants();
    }

    ~Lite3TestPolicyRunnerONNX() override = default;

    void DisplayPolicyInfo() override {
        std::cout << "ONNX policy: " << policy_name_ << "\n"
                  << "path: " << model_path_ << "\n"
                  << "obs_dim: " << kObsDim << ", act_dim: " << kActDim << "\n";
    }

    void OnEnter() override {
        run_cnt_ = 0;
        current_obs_.setZero(kObsDim);
        history_frames_.clear();
        pos_hist_.clear();
        vel_hist_.clear();
        tgt_hist_.clear();
        last_action_offset_.setZero(kActDim);
        SeedHistoryWithZeros();
        debug_dump_quota_ = ParseDebugQuota();
        std::cout << "[ONNX ENTER] History cleared. PolicyRunner entered: "
                  << policy_name_ << std::endl;
    }

    RobotAction GetRobotAction(const RobotBasicState& ro) override {
        // Build current 117-D observation frame (training order)
        VecXf joint_pos_policy(kActDim);
        VecXf joint_vel_policy(kActDim);
        MapRobotStateToPolicyOrder(ro, joint_pos_policy, joint_vel_policy);
        Vec3f cmd = ro.cmd_vel_normlized;
        SaturateVec3(cmd, -1.f, 1.f);

        // Match training's (legacy) quaternion interpretation for base_rpy.
        // Training code reads root_quat_w as if it were [x,y,z,w] (it is [w,x,y,z]),
        // so we intentionally mirror that behavior for parity.
        Vec3f base_rpy = ComputeTrainingRpyFromRotMat(ro.base_rot_mat);
        Vec3f projected_gravity = ro.projected_gravity;
        // Training uses base-frame angular velocity (quat_rotate_inverse).
        // Interfaces already provide body-frame IMU omega, so do NOT rotate again.
        Vec3f body_omega = ro.base_omega;
        Vec3f omega_world = ro.base_rot_mat * body_omega;

        if (pos_hist_.empty()) {
            SeedWithinFrameHistoriesWithCurrentJointState(joint_pos_policy, joint_vel_policy);
        }

        BuildCurrentObservation(cmd, base_rpy, body_omega,
                                joint_pos_policy, joint_vel_policy);
        // Match training: observations are clipped before being fed into HistoryWrapper/policy.
        current_obs_ = current_obs_.array().max(-kObsClip).min(kObsClip).matrix();

        // Update 40×117 history buffer (HistoryWrapper behaviour: oldest first).
        // Training play does two history appends before the first policy step, so we
        // mirror that by pushing the current obs twice on the very first call.
        const bool first_hist_push = (run_cnt_ == 0);
        history_frames_.push_back(current_obs_);
        if (first_hist_push) {
            history_frames_.push_back(current_obs_);
        }
        while (static_cast<int>(history_frames_.size()) > kHistoryLen) {
            history_frames_.pop_front();
        }

        // Flatten current obs + history for ONNX
        VecXf input_flat(kTotalInputDim);
        input_flat.segment(0, kObsDim) = current_obs_;
        int offset = kObsDim;
        for (const auto& frame : history_frames_) {
            input_flat.segment(offset, kObsDim) = frame;
            offset += kObsDim;
        }

        std::array<int64_t, 2> input_shape{1, kTotalInputDim};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_flat.data(), kTotalInputDim,
            input_shape.data(), input_shape.size());

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), &input_tensor, 1,
            output_names_.data(), 1);

        float* act_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> act_map(act_data, kActDim);
        action_raw_ = act_map;
        action_raw_ = action_raw_.array().max(-kActionClip).min(kActionClip).matrix();
        last_action_raw_ = action_raw_;

        VecXf policy_action_offset = action_raw_ * kTrainingActionScale;
        last_action_offset_ = policy_action_offset;

        VecXf robot_pd_target(kActDim);
        for (int i = 0; i < kActDim; ++i) {
            const int idx_policy = policy2robot_idx_[i];
            robot_pd_target(i) = dof_pos_default_robot_(i)
                               + policy_action_offset(idx_policy) * action_scale_robot_[i];
        }

        ra_.goal_joint_pos = robot_pd_target;
        ra_.goal_joint_vel = VecXf::Zero(kActDim);
        ra_.tau_ff         = VecXf::Zero(kActDim);
        ra_.kp             = kp_;
        ra_.kd             = kd_;

        DumpDebugIfRequested(
            input_flat,
            action_raw_,
            cmd,
            base_rpy,
            projected_gravity,
            body_omega,
            omega_world,
            ro.base_rot_mat,
            joint_pos_policy,
            joint_vel_policy,
            policy_action_offset,
            robot_pd_target
        );
        // Match training buffers: shift histories AFTER producing the current observation/action.
        UpdateWithinFrameHistories(joint_pos_policy, joint_vel_policy);
        ++run_cnt_;
        return ra_;
    }

private:
    static constexpr int   kObsDim        = 117;
    static constexpr int   kHistoryLen    = 40;
    static constexpr int   kTotalInputDim = kObsDim * (1 + kHistoryLen);
    static constexpr int   kActDim        = 12;
    static constexpr float kTrainingActionScale = 0.25f;
    static constexpr float kDofVelScale   = 0.1f;
    static constexpr float kActionClip    = 12.0f;
    static constexpr float kObsClip       = 100.0f;

    std::string       model_path_;
    Ort::Env          env_;
    Ort::SessionOptions session_options_;
    Ort::Session      session_{nullptr};
    Ort::MemoryInfo   memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    VecXf dof_pos_default_policy_;
    VecXf dof_pos_default_robot_;
    VecXf kp_;
    VecXf kd_;
    std::vector<int>  robot2policy_idx_;
    std::vector<int>  policy2robot_idx_;
    std::array<float, kActDim> action_scale_robot_{{
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f}};
    std::array<float, kActDim> joint_limit_lower_{};
    std::array<float, kActDim> joint_limit_upper_{};
    std::array<float, kActDim> effort_limit_{};

    VecXf current_obs_ = VecXf::Zero(kObsDim);
    std::deque<VecXf> history_frames_;
    std::deque<VecXf> pos_hist_;
    std::deque<VecXf> vel_hist_;
    std::deque<VecXf> tgt_hist_;

    VecXf action_raw_ = VecXf::Zero(kActDim);
    VecXf last_action_raw_ = VecXf::Zero(kActDim);
    VecXf last_action_offset_ = VecXf::Zero(kActDim);

    RobotAction ra_;
    int debug_dump_quota_ = 0;

    void InitSession() {
        // Allow overriding model path without recompiling.
        // - Absolute path: used as-is
        // - Relative path: resolved relative to current working directory (typically build/)
        const char* env_model = std::getenv("LITE3_POLICY_ONNX");
        if (env_model && *env_model != '\0') {
            std::filesystem::path p(env_model);
            model_path_ = p.is_absolute() ? p.string() : (std::filesystem::path(GetAbsPath()) / p).string();
        } else {
            model_path_ = GetAbsPath() + "/../policy/ppo/policy.onnx";
        }
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
        input_names_.push_back("obs");
        output_names_.push_back("action");
        LogModelInfo();
        // Match training control.decimation (TwoLegStandCfg.control.decimation = 4).
        decimation_ = 4;
        if (const char* dec_env = std::getenv("LITE3_POLICY_DECIMATION")) {
            const int parsed = std::atoi(dec_env);
            if (parsed > 0) {
                decimation_ = parsed;
            }
        }
        if (const char* dt_env = std::getenv("LITE3_MUJOCO_DT")) {
            char* endptr = nullptr;
            const double parsed = std::strtod(dt_env, &endptr);
            if (endptr != dt_env && std::isfinite(parsed) && parsed > 0.0) {
                const double ctrl_dt = parsed * static_cast<double>(decimation_);
                std::cout << "[ONNX] decimation=" << decimation_
                          << ", control_dt=" << ctrl_dt << "s (from LITE3_MUJOCO_DT)" << std::endl;
            }
        }
    }

    void InitRobotConstants() {
        // Match the training default_joint_angles used in TwoLegStandCfg.
        // Match training init_state joint_pos in Lite3TwoLegStandEnvCfg (base_env_cfg.py).
        dof_pos_default_policy_.resize(kActDim);
        dof_pos_default_policy_ <<
            -0.015f,  0.016f, -0.022f,  0.022f,  // HipX: FL, FR, HL, HR
            -0.770f, -0.770f, -0.770f, -0.770f,  // HipY: FL, FR, HL, HR
             1.540f,  1.540f,  1.550f,  1.550f;  // Knee: FL, FR, HL, HR

        kp_ = 20.f * VecXf::Ones(kActDim);
        kd_ =  0.7f * VecXf::Ones(kActDim);

        const std::vector<std::string> robot_order{
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};
        // Training observation order (joint_names=".*", preserve_order=True) is grouped by joint type.
        const std::vector<std::string> policy_order{
            "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
            "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
            "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint"};
        robot2policy_idx_ = BuildPermutation(robot_order, policy_order);
        policy2robot_idx_ = InvertPermutation(robot2policy_idx_);

        dof_pos_default_robot_ = VecXf::Zero(kActDim);
        for (int i = 0; i < kActDim; ++i) {
            const int idx_policy = policy2robot_idx_[i];
            dof_pos_default_robot_(i) = dof_pos_default_policy_(idx_policy);
        }

        // Joint limits from MJCF (radians), with soft limit factor to mirror training.
        constexpr float kSoft = 0.99f;
        // Order: FL_HipX, FL_HipY, FL_Knee, FR_HipX, FR_HipY, FR_Knee, HL_HipX, HL_HipY, HL_Knee, HR_HipX, HR_HipY, HR_Knee
        const std::array<float, kActDim> lower = {{
            -0.523f, -2.67f, 0.524f,
            -0.523f, -2.67f, 0.524f,
            -0.523f, -2.67f, 0.524f,
            -0.523f, -2.67f, 0.524f
        }};
        const std::array<float, kActDim> upper = {{
             0.523f,  0.314f, 2.792f,
             0.523f,  0.314f, 2.792f,
             0.523f,  0.314f, 2.792f,
             0.523f,  0.314f, 2.792f
        }};
        for (int i = 0; i < kActDim; ++i) {
            joint_limit_lower_[i] = lower[i] * kSoft;
            joint_limit_upper_[i] = upper[i] * kSoft;
        }
        // Effort limits from training actuators (Hip 24, Knee 36).
        effort_limit_ = {{
            24.f, 24.f, 36.f,
            24.f, 24.f, 36.f,
            24.f, 24.f, 36.f,
            24.f, 24.f, 36.f
        }};
    }

    static std::vector<int> BuildPermutation(const std::vector<std::string>& from,
                                             const std::vector<std::string>& to) {
        std::unordered_map<std::string, int> index;
        for (int i = 0; i < static_cast<int>(from.size()); ++i) {
            index[from[i]] = i;
        }
        std::vector<int> perm;
        perm.reserve(to.size());
        for (const auto& name : to) {
            auto it = index.find(name);
            perm.push_back(it != index.end() ? it->second : 0);
        }
        return perm;
    }

    static std::vector<int> InvertPermutation(const std::vector<int>& perm) {
        std::vector<int> inv(perm.size(), 0);
        for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
            int j = perm[i];
            if (j >= 0 && j < static_cast<int>(perm.size())) {
                inv[j] = i;
            }
        }
        return inv;
    }

    static Vec3f ComputeTrainingRpyFromRotMat(const Mat3f& rot_mat) {
        // Convert rotation matrix to quaternion (w, x, y, z).
        const float tr = rot_mat.trace();
        float w, x, y, z;
        if (tr > 0.0f) {
            const float S = std::sqrt(tr + 1.0f) * 2.0f;
            w = 0.25f * S;
            x = (rot_mat(2, 1) - rot_mat(1, 2)) / S;
            y = (rot_mat(0, 2) - rot_mat(2, 0)) / S;
            z = (rot_mat(1, 0) - rot_mat(0, 1)) / S;
        } else if ((rot_mat(0, 0) > rot_mat(1, 1)) && (rot_mat(0, 0) > rot_mat(2, 2))) {
            const float S = std::sqrt(1.0f + rot_mat(0, 0) - rot_mat(1, 1) - rot_mat(2, 2)) * 2.0f;
            w = (rot_mat(2, 1) - rot_mat(1, 2)) / S;
            x = 0.25f * S;
            y = (rot_mat(0, 1) + rot_mat(1, 0)) / S;
            z = (rot_mat(0, 2) + rot_mat(2, 0)) / S;
        } else if (rot_mat(1, 1) > rot_mat(2, 2)) {
            const float S = std::sqrt(1.0f + rot_mat(1, 1) - rot_mat(0, 0) - rot_mat(2, 2)) * 2.0f;
            w = (rot_mat(0, 2) - rot_mat(2, 0)) / S;
            x = (rot_mat(0, 1) + rot_mat(1, 0)) / S;
            y = 0.25f * S;
            z = (rot_mat(1, 2) + rot_mat(2, 1)) / S;
        } else {
            const float S = std::sqrt(1.0f + rot_mat(2, 2) - rot_mat(0, 0) - rot_mat(1, 1)) * 2.0f;
            w = (rot_mat(1, 0) - rot_mat(0, 1)) / S;
            x = (rot_mat(0, 2) + rot_mat(2, 0)) / S;
            y = (rot_mat(1, 2) + rot_mat(2, 1)) / S;
            z = 0.25f * S;
        }

        // Training code treats quaternion as [x, y, z, w] even though source is [w, x, y, z].
        const float qx = w;
        const float qy = x;
        const float qz = y;
        const float qw = z;

        const float sinr_cosp = 2.0f * (qw * qx + qy * qz);
        const float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
        float roll = std::atan2(sinr_cosp, cosr_cosp);

        float sinp = 2.0f * (qw * qy - qz * qx);
        sinp = std::min(1.0f, std::max(-1.0f, sinp));
        const float pitch = std::asin(sinp);

        const float siny_cosp = 2.0f * (qw * qz + qx * qy);
        const float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
        const float yaw = std::atan2(siny_cosp, cosy_cosp);

        // Training roll stays near -pi (due to quaternion mis-interpretation). Match that wrap.
        if (roll > 0.0f) {
            roll -= 2.0f * static_cast<float>(M_PI);
        }

        return Vec3f(roll, pitch, yaw);
    }

    void SeedHistoryWithZeros() {
        VecXf zero = VecXf::Zero(kObsDim);
        for (int i = 0; i < kHistoryLen; ++i) {
            history_frames_.push_back(zero);
        }
    }

    void SeedWithinFrameHistoriesWithCurrentJointState(const VecXf& joint_pos,
                                                       const VecXf& joint_vel) {
        pos_hist_.clear();
        vel_hist_.clear();
        tgt_hist_.clear();

        // Training warm-up fills history with standup/joint-damping data. To better match
        // that distribution, seed pos/vel history near zero and action history near -default.
        const char* seed_env = std::getenv("LITE3_HISTORY_SEED_CURRENT");
        const bool seed_current = (seed_env && std::atoi(seed_env) != 0);

        if (seed_current) {
            for (int i = 0; i < 3; ++i) pos_hist_.push_back(joint_pos);
            for (int i = 0; i < 2; ++i) vel_hist_.push_back(joint_vel);
            VecXf init_target = joint_pos - dof_pos_default_policy_;
            for (int i = 0; i < 2; ++i) tgt_hist_.push_back(init_target);
            last_action_offset_ = init_target;
            last_action_raw_ = init_target;
            return;
        }

        VecXf zero = VecXf::Zero(kActDim);
        for (int i = 0; i < 3; ++i) pos_hist_.push_back(zero);
        for (int i = 0; i < 2; ++i) vel_hist_.push_back(zero);
        VecXf init_target = -dof_pos_default_policy_;
        for (int i = 0; i < 2; ++i) tgt_hist_.push_back(init_target);
        last_action_offset_ = init_target;
        last_action_raw_ = init_target;
    }

    void UpdateWithinFrameHistories(const VecXf& joint_pos_policy,
                                    const VecXf& joint_vel_policy) {
        if (pos_hist_.empty()) {
            SeedWithinFrameHistoriesWithCurrentJointState(joint_pos_policy, joint_vel_policy);
        }

        pos_hist_.push_back(joint_pos_policy);
        if (static_cast<int>(pos_hist_.size()) > 3) pos_hist_.pop_front();

        vel_hist_.push_back(joint_vel_policy);
        if (static_cast<int>(vel_hist_.size()) > 2) vel_hist_.pop_front();

        tgt_hist_.push_back(last_action_raw_);
        if (static_cast<int>(tgt_hist_.size()) > 2) tgt_hist_.pop_front();
    }

    void MapRobotStateToPolicyOrder(const RobotBasicState& ro,
                                    VecXf& joint_pos_policy,
                                    VecXf& joint_vel_policy) const {
        for (int i = 0; i < kActDim; ++i) {
            const int idx = robot2policy_idx_[i];
            joint_pos_policy(i) = ro.joint_pos(idx);
            joint_vel_policy(i) = ro.joint_vel(idx) * kDofVelScale;
        }
    }

    static void SaturateVec3(Vec3f& v, float low, float high) {
        for (int i = 0; i < 3; ++i) {
            v(i) = std::min(std::max(v(i), low), high);
        }
    }

    void BuildCurrentObservation(const Vec3f& cmd,
                                 const Vec3f& base_rpy,
                                 const Vec3f& body_omega,
                                 const VecXf& joint_pos_policy,
                                 const VecXf& joint_vel_policy) {
        VecXf pos_hist_flat = VecXf::Zero(36);
        VecXf vel_hist_flat = VecXf::Zero(24);
        VecXf tgt_hist_flat = VecXf::Zero(24);

        int idx = 0;
        for (const auto& v : pos_hist_) {
            pos_hist_flat.segment(idx, kActDim) = v;
            idx += kActDim;
        }
        idx = 0;
        for (const auto& v : vel_hist_) {
            vel_hist_flat.segment(idx, kActDim) = v;
            idx += kActDim;
        }
        idx = 0;
        for (const auto& v : tgt_hist_) {
            tgt_hist_flat.segment(idx, kActDim) = v;
            idx += kActDim;
        }

        current_obs_.head(3)        = cmd;
        current_obs_.segment(3, 3)  = base_rpy;
        current_obs_.segment(6, 3)  = body_omega;
        current_obs_.segment(9, 12) = joint_pos_policy;
        current_obs_.segment(21,12) = joint_vel_policy;
        current_obs_.segment(33,36) = pos_hist_flat;
        current_obs_.segment(69,24) = vel_hist_flat;
        current_obs_.segment(93,24) = tgt_hist_flat;
    }

    int ParseDebugQuota() const {
        const char* env = std::getenv("LITE3_DEBUG_DUMPS");
        if (!env) return 5;
        try {
            return std::max(0, std::stoi(env));
        } catch (...) {
            return 5;
        }
    }

    static uint64_t Fnv1a64File(const std::string& path, size_t* out_size) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            if (out_size) *out_size = 0;
            return 0;
        }
        const uint64_t kOffset = 14695981039346656037ULL;
        const uint64_t kPrime = 1099511628211ULL;
        uint64_t hash = kOffset;
        size_t total = 0;
        char buf[8192];
        while (ifs.good()) {
            ifs.read(buf, sizeof(buf));
            std::streamsize n = ifs.gcount();
            for (std::streamsize i = 0; i < n; ++i) {
                hash ^= static_cast<unsigned char>(buf[i]);
                hash *= kPrime;
            }
            total += static_cast<size_t>(n);
        }
        if (out_size) *out_size = total;
        return hash;
    }

    static std::string ShapeToString(const std::vector<int64_t>& shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) oss << ", ";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }

    void LogModelInfo() {
        std::cout << "[ONNX] model path: " << model_path_ << "\n";
        size_t fsize = 0;
        const uint64_t hash = Fnv1a64File(model_path_, &fsize);
        if (hash != 0) {
            std::cout << "[ONNX] file size: " << fsize << " bytes\n";
            std::cout << "[ONNX] fnv1a64: 0x" << std::hex << std::setw(16)
                      << std::setfill('0') << hash << std::dec << "\n";
        } else {
            std::cout << "[ONNX] file hash unavailable (read failed)\n";
        }

        Ort::AllocatorWithDefaultOptions alloc;
        const size_t num_inputs = session_.GetInputCount();
        const size_t num_outputs = session_.GetOutputCount();
        std::cout << "[ONNX] inputs: " << num_inputs << ", outputs: " << num_outputs << "\n";

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_.GetInputNameAllocated(i, alloc);
            auto type_info = session_.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            std::cout << "[ONNX] input[" << i << "] name=" << name.get()
                      << " shape=" << ShapeToString(shape) << "\n";
        }
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_.GetOutputNameAllocated(i, alloc);
            auto type_info = session_.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            std::cout << "[ONNX] output[" << i << "] name=" << name.get()
                      << " shape=" << ShapeToString(shape) << "\n";
        }
    }

    void DumpDebugIfRequested(const VecXf& input_flat,
                              const VecXf& policy_action_raw,
                              const Vec3f& cmd,
                              const Vec3f& base_rpy,
                              const Vec3f& projected_gravity,
                              const Vec3f& body_omega,
                              const Vec3f& omega_world,
                              const Mat3f& base_rot_mat,
                              const VecXf& joint_pos_policy,
                              const VecXf& joint_vel_policy,
                              const VecXf& action_offset,
                              const VecXf& target_joint_pos) {
        if (debug_dump_quota_ <= 0) return;
        const char* env_dir = std::getenv("LITE3_DEBUG_DUMP_DIR");
        std::vector<std::string> candidates;
        if (env_dir && *env_dir != '\0') {
            candidates.emplace_back(env_dir);
        }
        if (std::filesystem::exists("/workspace/rl_training_new")) {
            candidates.emplace_back("/workspace/rl_training_new/lite3_debug/deploy");
            candidates.emplace_back("/workspace/rl_training_new/debug_deploy");
        }
        candidates.emplace_back("/tmp/lite3_debug");

        std::string dump_root;
        for (const auto& candidate : candidates) {
            std::error_code ec;
            std::filesystem::create_directories(candidate, ec);
            if (!ec) {
                dump_root = candidate;
                break;
            }
        }
        if (dump_root.empty()) {
            std::cerr << "[DEBUG] Failed to create any debug dump directory; disabling dumps.\n";
            debug_dump_quota_ = 0;
            return;
        }
        const std::string fname = dump_root + "/debug_cpp_step" + std::to_string(run_cnt_) + ".txt";
        std::ofstream ofs(fname);
        if (!ofs.is_open()) {
            std::cerr << "[DEBUG] Failed to open " << fname << " for writing; disabling dumps.\n";
            debug_dump_quota_ = 0;
            return;
        }
        ofs << "cmd";
        for (int i = 0; i < 3; ++i) ofs << " " << cmd(i);
        ofs << "\nbase_rpy";
        for (int i = 0; i < 3; ++i) ofs << " " << base_rpy(i);
        ofs << "\nprojected_gravity";
        for (int i = 0; i < 3; ++i) ofs << " " << projected_gravity(i);
        ofs << "\nbody_omega";
        for (int i = 0; i < 3; ++i) ofs << " " << body_omega(i);
        ofs << "\nomega_world";
        for (int i = 0; i < 3; ++i) ofs << " " << omega_world(i);
        ofs << "\nbase_rot_mat";
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                ofs << " " << base_rot_mat(r, c);
            }
        }
        ofs << "\njoint_pos_policy";
        for (int i = 0; i < kActDim; ++i) ofs << " " << joint_pos_policy(i);
        ofs << "\njoint_vel_policy";
        for (int i = 0; i < kActDim; ++i) ofs << " " << joint_vel_policy(i);
        ofs << "\naction_raw";
        for (int i = 0; i < kActDim; ++i) ofs << " " << policy_action_raw(i);
        ofs << "\naction_offset";
        for (int i = 0; i < kActDim; ++i) ofs << " " << action_offset(i);
        ofs << "\ntarget_joint_pos";
        for (int i = 0; i < kActDim; ++i) ofs << " " << target_joint_pos(i);
        // Compute robot-order joint state for torque diagnostics.
        VecXf joint_pos_robot(kActDim);
        VecXf joint_vel_robot(kActDim);
        for (int i = 0; i < kActDim; ++i) {
            const int idx_policy = policy2robot_idx_[i];
            joint_pos_robot(i) = joint_pos_policy(idx_policy);
            joint_vel_robot(i) = joint_vel_policy(idx_policy);
        }
        VecXf target_joint_pos_clipped = target_joint_pos;
        for (int i = 0; i < kActDim; ++i) {
            target_joint_pos_clipped(i) = std::min(std::max(target_joint_pos_clipped(i),
                                                            joint_limit_lower_[i]),
                                                   joint_limit_upper_[i]);
        }
        VecXf pd_tau_raw = kp_.array() * (target_joint_pos - joint_pos_robot).array()
                         + kd_.array() * (VecXf::Zero(kActDim) - joint_vel_robot).array();
        VecXf pd_tau_clipped = pd_tau_raw;
        for (int i = 0; i < kActDim; ++i) {
            const float lim = effort_limit_[i];
            pd_tau_clipped(i) = std::min(std::max(pd_tau_clipped(i), -lim), lim);
        }
        ofs << "\njoint_limits_lower";
        for (int i = 0; i < kActDim; ++i) ofs << " " << joint_limit_lower_[i];
        ofs << "\njoint_limits_upper";
        for (int i = 0; i < kActDim; ++i) ofs << " " << joint_limit_upper_[i];
        ofs << "\neffort_limits";
        for (int i = 0; i < kActDim; ++i) ofs << " " << effort_limit_[i];
        ofs << "\ntarget_joint_pos_clipped";
        for (int i = 0; i < kActDim; ++i) ofs << " " << target_joint_pos_clipped(i);
        ofs << "\npd_tau_raw";
        for (int i = 0; i < kActDim; ++i) ofs << " " << pd_tau_raw(i);
        ofs << "\npd_tau_clipped";
        for (int i = 0; i < kActDim; ++i) ofs << " " << pd_tau_clipped(i);
        ofs << "\nobs_flat";
        for (int i = 0; i < kTotalInputDim; ++i) {
            ofs << " " << input_flat(i);
        }
        ofs << std::endl;
        --debug_dump_quota_;
    }
};

#endif // LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
