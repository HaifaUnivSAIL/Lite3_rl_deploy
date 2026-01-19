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

        Vec3f base_rpy = ro.base_rpy;
        // Training uses base-frame angular velocity (quat_rotate_inverse).
        // Interfaces already provide body-frame IMU omega, so do NOT rotate again.
        Vec3f body_omega = ro.base_omega;

        if (pos_hist_.empty()) {
            SeedWithinFrameHistoriesWithCurrentJointState(joint_pos_policy, joint_vel_policy);
        }

        BuildCurrentObservation(cmd, base_rpy, body_omega,
                                joint_pos_policy, joint_vel_policy);
        // Match training: observations are clipped before being fed into HistoryWrapper/policy.
        current_obs_ = current_obs_.array().max(-kObsClip).min(kObsClip).matrix();

        // Update 40×117 history buffer (HistoryWrapper behaviour: oldest first).
        // Start with zeroed frames (set in OnEnter), then push current obs and drop oldest.
        history_frames_.push_back(current_obs_);
        if (static_cast<int>(history_frames_.size()) > kHistoryLen) {
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

        DumpDebugIfRequested(input_flat, action_raw_);
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

    VecXf current_obs_ = VecXf::Zero(kObsDim);
    std::deque<VecXf> history_frames_;
    std::deque<VecXf> pos_hist_;
    std::deque<VecXf> vel_hist_;
    std::deque<VecXf> tgt_hist_;

    VecXf action_raw_ = VecXf::Zero(kActDim);
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
        // Match training control.decimation (TwoLegStandCfg.control.decimation = 4)
        decimation_ = 4;
    }

    void InitRobotConstants() {
        // Match the training default_joint_angles used in TwoLegStandCfg.
        dof_pos_default_policy_.resize(kActDim);
        dof_pos_default_policy_ <<
            -0.0154048f, -0.76697f,  1.53761f,   // FL_HipX, FL_HipY, FL_Knee
             0.0159887f, -0.768286f, 1.53636f,   // FR_HipX, FR_HipY, FR_Knee
            -0.0221317f, -0.765865f, 1.54788f,   // HL_HipX, HL_HipY, HL_Knee
             0.0224431f, -0.767203f, 1.54679f;   // HR_HipX, HR_HipY, HR_Knee
        dof_pos_default_robot_ = dof_pos_default_policy_;

        kp_ = 20.f * VecXf::Ones(kActDim);
        kd_ =  0.7f * VecXf::Ones(kActDim);

        const std::vector<std::string> order{
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};
        robot2policy_idx_ = BuildPermutation(order, order);
        policy2robot_idx_ = InvertPermutation(robot2policy_idx_);
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
        for (int i = 0; i < 3; ++i) pos_hist_.push_back(joint_pos);
        for (int i = 0; i < 2; ++i) vel_hist_.push_back(joint_vel);
        VecXf init_target = joint_pos - dof_pos_default_policy_;
        for (int i = 0; i < 2; ++i) tgt_hist_.push_back(init_target);
        last_action_offset_ = init_target;
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

        tgt_hist_.push_back(last_action_offset_);
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
        if (!env) return 0;
        try {
            return std::max(0, std::stoi(env));
        } catch (...) {
            return 0;
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
                              const VecXf& policy_action_raw) {
        if (debug_dump_quota_ <= 0) return;
        const std::string dump_root = GetAbsPath() + "/../../Lite3_rl_training/debug_training_obs/";
        std::filesystem::create_directories(dump_root);
        const std::string fname = dump_root + "debug_cpp_step" + std::to_string(run_cnt_) + ".txt";
        std::ofstream ofs(fname);
        if (!ofs.is_open()) {
            std::cerr << "[DEBUG] Failed to open " << fname << " for writing\n";
            return;
        }
        ofs << "obs_flat";
        for (int i = 0; i < kTotalInputDim; ++i) {
            ofs << " " << input_flat(i);
        }
        ofs << "\naction";
        for (int i = 0; i < kActDim; ++i) {
            ofs << " " << policy_action_raw(i);
        }
        ofs << std::endl;
        --debug_dump_quota_;
    }
};

#endif // LITE3_TEST_POLICY_RUNNER_ONNX_HPP_
