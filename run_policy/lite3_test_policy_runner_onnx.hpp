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
#include <iostream>
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
        pending_action_hist_.clear();
        last_action_offset_.setZero(kActDim);
        SeedHistoryWithZeros();
        LoadHistorySeedFromFileEnv();
        ResolveBaseRpySourceFromEnv();
        debug_dump_quota_ = ParseDebugQuota();
        std::cout << "[ONNX ENTER] History cleared. PolicyRunner entered: "
                  << policy_name_
                  << ", base_rpy_source=" << base_rpy_source_label_
                  << std::endl;
    }

    RobotAction GetRobotAction(const RobotBasicState& ro) override {
        // Build current 117-D observation frame (training order)
        VecXf joint_pos_policy(kActDim);
        VecXf joint_vel_policy(kActDim);
        MapRobotStateToPolicyOrder(ro, joint_pos_policy, joint_vel_policy);
        Vec3f cmd = ro.cmd_vel_normlized;
        SaturateVec3(cmd, -1.f, 1.f);

        // Build base_rpy according to selected deploy convention.
        const bool quat_valid = ro.base_quat.allFinite() && ro.base_quat.norm() > 1e-6f;
        const Vec4f quat_identity(1.f, 0.f, 0.f, 0.f);
        const bool quat_is_identity = quat_valid &&
            (ro.base_quat - quat_identity).cwiseAbs().maxCoeff() < 1e-6f;
        const bool use_rotmat_fallback = quat_is_identity && run_cnt_ == 0;
        const Vec3f base_rpy_interface = ro.base_rpy;
        const Vec3f base_rpy_from_quat_std = (quat_valid && !use_rotmat_fallback)
            ? ComputeRpyFromQuatWxyzStandard(ro.base_quat.normalized())
            : ComputeRpyFromRotMatStandard(ro.base_rot_mat);
        const Vec3f base_rpy_from_quat_legacy = (quat_valid && !use_rotmat_fallback)
            ? ComputeTrainingLegacyRpyFromQuatWxyz(ro.base_quat.normalized())
            : ComputeTrainingLegacyRpyFromRotMat(ro.base_rot_mat);

        Vec3f base_rpy = base_rpy_interface;
        if (base_rpy_source_ == BaseRpySourceMode::kQuatWxyzStandard) {
            base_rpy = base_rpy_from_quat_std;
        } else if (base_rpy_source_ == BaseRpySourceMode::kQuatLegacyXyzw) {
            base_rpy = base_rpy_from_quat_legacy;
        }

        // Keep first-step +/-pi canonicalization only for legacy mode fallback.
        if (base_rpy_source_ == BaseRpySourceMode::kQuatLegacyXyzw && run_cnt_ == 0) {
            base_rpy(0) = CanonicalizePiBranchNegative(base_rpy(0));
        }
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
        history_frames_.push_back(current_obs_);
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
            base_rpy_interface,
            base_rpy_from_quat_std,
            base_rpy_from_quat_legacy,
            projected_gravity,
            body_omega,
            ro.base_acc,
            omega_world,
            ro.base_quat,
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
    enum class BaseRpySourceMode {
        kInterfaceRpy = 0,
        kQuatWxyzStandard = 1,
        kQuatLegacyXyzw = 2,
    };

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
    std::deque<VecXf> pending_action_hist_;

    VecXf action_raw_ = VecXf::Zero(kActDim);
    VecXf last_action_raw_ = VecXf::Zero(kActDim);
    VecXf last_action_offset_ = VecXf::Zero(kActDim);

    RobotAction ra_;
    int debug_dump_quota_ = 0;
    bool action_hist_delayed_mode_ = false;
    std::string history_seed_mode_ = "training_parity";
    bool history_seed_file_loaded_ = false;
    bool history_seed_file_consumed_ = false;
    std::string history_seed_file_path_;
    std::vector<float> history_seed_pos_hist_;
    std::vector<float> history_seed_vel_hist_;
    std::vector<float> history_seed_act_hist_;
    std::vector<float> history_seed_obs_hist_;
    BaseRpySourceMode base_rpy_source_ = BaseRpySourceMode::kInterfaceRpy;
    std::string base_rpy_source_label_ = "interface_rpy";

    static int ParsePositiveIntEnv(const char* env_name, int fallback) {
        const char* env_val = std::getenv(env_name);
        if (!env_val) {
            return fallback;
        }
        char* endptr = nullptr;
        long parsed = std::strtol(env_val, &endptr, 10);
        if (endptr == env_val || !std::isfinite(static_cast<double>(parsed)) || parsed <= 0) {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    static double ParsePositiveDoubleEnv(const char* env_name, double fallback) {
        const char* env_val = std::getenv(env_name);
        if (!env_val) {
            return fallback;
        }
        char* endptr = nullptr;
        const double parsed = std::strtod(env_val, &endptr);
        if (endptr == env_val || !std::isfinite(parsed) || parsed <= 0.0) {
            return fallback;
        }
        return parsed;
    }

    void AssertTrainingDeployTimingParity(double deploy_sim_dt, int deploy_decimation) const {
        // Source-of-truth defaults from rl_training_new TwoLegStandEnvCfg:
        //   decimation=4, sim.dt=0.005 -> control_dt=0.02s
        const int training_decimation =
            ParsePositiveIntEnv("LITE3_TRAINING_DECIMATION", 4);
        const double training_sim_dt =
            ParsePositiveDoubleEnv("LITE3_TRAINING_SIM_DT", 0.005);
        const double tol =
            ParsePositiveDoubleEnv("LITE3_TIMING_ASSERT_TOL", 1.0e-6);

        const double training_control_dt =
            training_sim_dt * static_cast<double>(training_decimation);
        const double deploy_control_dt =
            deploy_sim_dt * static_cast<double>(deploy_decimation);
        const int expected_deploy_decimation =
            std::max(1, static_cast<int>(std::lround(training_control_dt / deploy_sim_dt)));

        const bool decimation_match = (deploy_decimation == expected_deploy_decimation);
        const bool control_dt_match = std::fabs(deploy_control_dt - training_control_dt) <= tol;

        std::cout << "[ONNX] timing parity check: "
                  << "training(sim_dt=" << training_sim_dt
                  << ", decimation=" << training_decimation
                  << ", control_dt=" << training_control_dt << "s) vs "
                  << "deploy(sim_dt=" << deploy_sim_dt
                  << ", decimation=" << deploy_decimation
                  << ", control_dt=" << deploy_control_dt << "s)"
                  << std::endl;

        if (!decimation_match || !control_dt_match) {
            std::cerr << "[ONNX][FATAL] training/deploy timing mismatch. "
                      << "expected deploy decimation=" << expected_deploy_decimation
                      << ", got=" << deploy_decimation
                      << "; expected control_dt=" << training_control_dt
                      << "s, got=" << deploy_control_dt << "s. "
                      << "Override with LITE3_TRAINING_DECIMATION / LITE3_TRAINING_SIM_DT "
                      << "only if you intentionally changed training timing."
                      << std::endl;
            std::exit(2);
        }
    }

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
        // Match training control period by default (target control_dt = 0.02s).
        // Backend-specific default dt:
        // - legacy UDP sim path (USE_PYBULLET): 1kHz interface loop (dt=0.001)
        // - MuJoCo C++ path: dt=0.005 unless overridden
        // - hardware path: keep conservative 1kHz assumption (dt=0.001)
        double sim_dt = 0.005;
        const char* sim_dt_source = "default_mujoco_cpp";
#if defined(BUILD_SIMULATION) && defined(USE_PYBULLET)
        sim_dt = 0.001;
        sim_dt_source = "default_legacy_udp";
#elif !defined(BUILD_SIMULATION)
        sim_dt = 0.001;
        sim_dt_source = "default_hardware";
#endif
        if (const char* dt_env = std::getenv("LITE3_MUJOCO_DT")) {
            char* endptr = nullptr;
            const double parsed = std::strtod(dt_env, &endptr);
            if (endptr != dt_env && std::isfinite(parsed) && parsed > 0.0) {
                sim_dt = parsed;
                sim_dt_source = "env:LITE3_MUJOCO_DT";
            }
        }
        decimation_ = std::max(1, static_cast<int>(std::lround(0.02 / sim_dt)));
        if (const char* dec_env = std::getenv("LITE3_POLICY_DECIMATION")) {
            const int parsed = std::atoi(dec_env);
            if (parsed > 0) {
                decimation_ = parsed;
            }
        }
        AssertTrainingDeployTimingParity(sim_dt, decimation_);
        const double ctrl_dt = sim_dt * static_cast<double>(decimation_);
        std::cout << "[ONNX] sim_dt=" << sim_dt
                  << ", decimation=" << decimation_
                  << ", control_dt=" << ctrl_dt << "s"
                  << ", sim_dt_source=" << sim_dt_source << std::endl;
    }

    void InitRobotConstants() {
        // Match the training default_joint_angles used in TwoLegStandCfg.
        // Match training init_state joint_pos in Lite3TwoLegStandEnvCfg (base_env_cfg.py).
        dof_pos_default_policy_.resize(kActDim);
        dof_pos_default_policy_ <<
            -0.0154048f,  0.0159887f, -0.0221317f,  0.0224431f,  // HipX: FL, FR, HL, HR
            -0.7669700f, -0.7682860f, -0.7658650f, -0.7672030f,  // HipY: FL, FR, HL, HR
             1.5376100f,  1.5363600f,  1.5478800f,  1.5467900f;  // Knee: FL, FR, HL, HR

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

    static Vec3f ComputeRpyFromQuatWxyzStandard(const Vec4f& quat_wxyz) {
        const float w = quat_wxyz(0);
        const float x = quat_wxyz(1);
        const float y = quat_wxyz(2);
        const float z = quat_wxyz(3);

        const float sinr_cosp = 2.0f * (w * x + y * z);
        const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        const float roll = std::atan2(sinr_cosp, cosr_cosp);

        float sinp = 2.0f * (w * y - z * x);
        sinp = std::max(-1.0f, std::min(1.0f, sinp));
        const float pitch = std::asin(sinp);

        const float siny_cosp = 2.0f * (w * z + x * y);
        const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        const float yaw = std::atan2(siny_cosp, cosy_cosp);

        return Vec3f(roll, pitch, yaw);
    }

    static Vec3f ComputeRpyFromRotMatStandard(const Mat3f& rot_mat) {
        Eigen::Quaternionf q(rot_mat);
        q.normalize();
        Vec4f quat_wxyz;
        quat_wxyz << q.w(), q.x(), q.y(), q.z();
        return ComputeRpyFromQuatWxyzStandard(quat_wxyz);
    }

    static Vec3f ComputeTrainingLegacyRpyFromQuatWxyz(const Vec4f& quat_wxyz) {
        const float qw = quat_wxyz(0);
        const float qx = quat_wxyz(1);
        const float qy = quat_wxyz(2);
        const float qz = quat_wxyz(3);

        // Training code treats root_quat_w as XYZW, while source is WXYZ.
        const float w = qz;
        const float x = qw;
        const float y = qx;
        const float z = qy;

        const float sinr_cosp = 2.0f * (w * x + y * z);
        const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        const float roll = std::atan2(sinr_cosp, cosr_cosp);

        float sinp = 2.0f * (w * y - z * x);
        sinp = std::max(-1.0f, std::min(1.0f, sinp));
        const float pitch = std::asin(sinp);

        const float siny_cosp = 2.0f * (w * z + x * y);
        const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        const float yaw = std::atan2(siny_cosp, cosy_cosp);

        return Vec3f(roll, pitch, yaw);
    }

    static Vec3f ComputeTrainingLegacyRpyFromRotMat(const Mat3f& rot_mat) {
        Eigen::Quaternionf q(rot_mat);
        q.normalize();
        Vec4f quat_wxyz;
        quat_wxyz << q.w(), q.x(), q.y(), q.z();
        return ComputeTrainingLegacyRpyFromQuatWxyz(quat_wxyz);
    }

    static float CanonicalizePiBranchNegative(float angle_rad) {
        // Training/deploy parity issue: the legacy quaternion->RPY path can flip between +pi and -pi
        // for essentially the same attitude due to tiny yaw sign differences near identity.
        constexpr float kPi = static_cast<float>(M_PI);
        constexpr float kNearPiEps = 0.25f;
        if (std::abs(std::abs(angle_rad) - kPi) < kNearPiEps) {
            return -std::abs(angle_rad);
        }
        return angle_rad;
    }

    void SeedHistoryWithZeros() {
        VecXf zero = VecXf::Zero(kObsDim);
        for (int i = 0; i < kHistoryLen; ++i) {
            history_frames_.push_back(zero);
        }
    }

    bool LoadHistorySeedFromFileEnv() {
        history_seed_file_loaded_ = false;
        history_seed_file_consumed_ = false;
        history_seed_file_path_.clear();
        history_seed_pos_hist_.clear();
        history_seed_vel_hist_.clear();
        history_seed_act_hist_.clear();
        history_seed_obs_hist_.clear();

        const char* env_path = std::getenv("LITE3_HISTORY_SEED_FILE");
        if (!(env_path && *env_path != '\0')) {
            return false;
        }
        history_seed_file_path_ = env_path;

        std::ifstream ifs(history_seed_file_path_);
        if (!ifs.is_open()) {
            std::cerr << "[ONNX] history seed file open failed: " << history_seed_file_path_ << "\n";
            return false;
        }

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            if (key.empty()) {
                continue;
            }

            std::vector<float>* target = nullptr;
            if (key == "joint_pos_history" || key == "pos_hist") {
                target = &history_seed_pos_hist_;
            } else if (key == "joint_vel_history" || key == "vel_hist") {
                target = &history_seed_vel_hist_;
            } else if (key == "action_history" || key == "tgt_hist") {
                target = &history_seed_act_hist_;
            } else if (key == "obs_history") {
                target = &history_seed_obs_hist_;
            } else {
                continue;
            }

            target->clear();
            float value = 0.0f;
            while (iss >> value) {
                target->push_back(value);
            }
        }

        const bool ok_pos = history_seed_pos_hist_.size() == 36;
        const bool ok_vel = history_seed_vel_hist_.size() == 24;
        const bool ok_act = history_seed_act_hist_.size() == 24;
        const bool ok_obs = history_seed_obs_hist_.empty() || history_seed_obs_hist_.size() == (kHistoryLen * kObsDim);
        if (!(ok_pos && ok_vel && ok_act && ok_obs)) {
            std::cerr << "[ONNX] history seed file invalid sizes. "
                      << "pos=" << history_seed_pos_hist_.size()
                      << " vel=" << history_seed_vel_hist_.size()
                      << " act=" << history_seed_act_hist_.size()
                      << " obs=" << history_seed_obs_hist_.size() << "\n";
            history_seed_pos_hist_.clear();
            history_seed_vel_hist_.clear();
            history_seed_act_hist_.clear();
            history_seed_obs_hist_.clear();
            return false;
        }

        history_seed_file_loaded_ = true;
        std::cout << "[ONNX] history seed file loaded: " << history_seed_file_path_
                  << " (obs_history=" << (history_seed_obs_hist_.empty() ? "missing" : "present") << ")\n";
        return true;
    }

    bool ApplyHistorySeedFromFile() {
        if (!history_seed_file_loaded_ || history_seed_file_consumed_) {
            return false;
        }

        pos_hist_.clear();
        vel_hist_.clear();
        tgt_hist_.clear();
        pending_action_hist_.clear();

        for (int frame = 0; frame < 3; ++frame) {
            VecXf v(kActDim);
            for (int j = 0; j < kActDim; ++j) {
                v(j) = history_seed_pos_hist_[frame * kActDim + j];
            }
            pos_hist_.push_back(v);
        }
        for (int frame = 0; frame < 2; ++frame) {
            VecXf v(kActDim);
            for (int j = 0; j < kActDim; ++j) {
                v(j) = history_seed_vel_hist_[frame * kActDim + j];
            }
            vel_hist_.push_back(v);
        }
        for (int frame = 0; frame < 2; ++frame) {
            VecXf v(kActDim);
            for (int j = 0; j < kActDim; ++j) {
                v(j) = history_seed_act_hist_[frame * kActDim + j];
            }
            tgt_hist_.push_back(v);
        }

        if (!history_seed_obs_hist_.empty()) {
            history_frames_.clear();
            for (int frame = 0; frame < kHistoryLen; ++frame) {
                VecXf v(kObsDim);
                for (int j = 0; j < kObsDim; ++j) {
                    v(j) = history_seed_obs_hist_[frame * kObsDim + j];
                }
                history_frames_.push_back(v);
            }
        }

        action_hist_delayed_mode_ = false;
        history_seed_mode_ = "replay_file";
        last_action_raw_ = tgt_hist_.back();
        last_action_offset_ = last_action_raw_;
        history_seed_file_consumed_ = true;
        std::cout << "[ONNX] history replay applied from file on reset.\n";
        return true;
    }

    void SeedWithinFrameHistoriesWithCurrentJointState(const VecXf& joint_pos,
                                                       const VecXf& joint_vel) {
        pos_hist_.clear();
        vel_hist_.clear();
        tgt_hist_.clear();
        pending_action_hist_.clear();
        if (ApplyHistorySeedFromFile()) {
            return;
        }
        const VecXf joint_pos_rel = joint_pos - dof_pos_default_policy_;
        const VecXf reset_action = joint_pos_rel;
        const char* mode_env = std::getenv("LITE3_HISTORY_SEED_MODE");
        const std::string mode = (mode_env && *mode_env != '\0') ? std::string(mode_env) : "training_parity";
        history_seed_mode_ = mode;
        action_hist_delayed_mode_ = false;

        if (mode == "current_rel") {
            std::cout << "[ONNX] history seed mode: current_rel (pos=rel, vel=current, tgt=rel)" << std::endl;
            for (int i = 0; i < 3; ++i) pos_hist_.push_back(joint_pos_rel);
            for (int i = 0; i < 2; ++i) vel_hist_.push_back(joint_vel);
            for (int i = 0; i < 2; ++i) tgt_hist_.push_back(joint_pos_rel);
            last_action_offset_ = joint_pos_rel;
            last_action_raw_ = joint_pos_rel;
            return;
        }

        if (mode == "standup_like") {
            VecXf zero = VecXf::Zero(kActDim);
            std::cout << "[ONNX] history seed mode: standup_like (pos=0, vel=current, tgt=-default)" << std::endl;
            for (int i = 0; i < 3; ++i) pos_hist_.push_back(zero);
            for (int i = 0; i < 2; ++i) vel_hist_.push_back(joint_vel);
            VecXf init_target = -dof_pos_default_policy_;
            for (int i = 0; i < 2; ++i) tgt_hist_.push_back(init_target);
            last_action_offset_ = init_target;
            last_action_raw_ = init_target;
            return;
        }

        // Default path: match training observation history semantics:
        // - pos history uses absolute joint positions,
        // - vel history uses current joint velocity,
        // - action history in play parity starts from zeros and updates with a one-step delay:
        //   [0,0] -> [0,a0] -> [a0,a1] ...
        std::cout << "[ONNX] history seed mode: training_parity (pos=abs, vel=current, tgt=zero+delay1)" << std::endl;
        action_hist_delayed_mode_ = true;
        for (int i = 0; i < 3; ++i) pos_hist_.push_back(joint_pos);
        for (int i = 0; i < 2; ++i) vel_hist_.push_back(joint_vel);
        for (int i = 0; i < 2; ++i) tgt_hist_.push_back(VecXf::Zero(kActDim));
        last_action_offset_ = reset_action;
        last_action_raw_ = reset_action;
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

        // action_history stores (t-2, t-1).
        // For training parity, keep one-step delayed insertion:
        // step0 obs=[0,0], step1 obs=[0,0], step2 obs=[0,a0], step3 obs=[a0,a1], ...
        if (action_hist_delayed_mode_) {
            pending_action_hist_.push_back(last_action_raw_);
            if (static_cast<int>(pending_action_hist_.size()) > 1) {
                tgt_hist_.push_back(pending_action_hist_.front());
                pending_action_hist_.pop_front();
                if (static_cast<int>(tgt_hist_.size()) > 2) tgt_hist_.pop_front();
            }
        } else {
            tgt_hist_.push_back(last_action_raw_);
            if (static_cast<int>(tgt_hist_.size()) > 2) tgt_hist_.pop_front();
        }
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

    static const char* BaseRpySourceToString(BaseRpySourceMode mode) {
        switch (mode) {
            case BaseRpySourceMode::kInterfaceRpy:
                return "interface_rpy";
            case BaseRpySourceMode::kQuatWxyzStandard:
                return "quat_wxyz_standard";
            case BaseRpySourceMode::kQuatLegacyXyzw:
                return "quat_legacy_xyzw";
            default:
                return "interface_rpy";
        }
    }

    void ResolveBaseRpySourceFromEnv() {
        const char* env = std::getenv("LITE3_BASE_RPY_SOURCE");
        BaseRpySourceMode mode = BaseRpySourceMode::kInterfaceRpy;
        if (env && *env != '\0') {
            const std::string requested(env);
            if (requested == "interface_rpy" || requested == "imu_rpy") {
                mode = BaseRpySourceMode::kInterfaceRpy;
            } else if (requested == "quat_wxyz_standard" || requested == "quat_standard") {
                mode = BaseRpySourceMode::kQuatWxyzStandard;
            } else if (requested == "quat_legacy_xyzw" || requested == "legacy_xyzw") {
                mode = BaseRpySourceMode::kQuatLegacyXyzw;
            } else {
                std::cerr << "[ONNX] Unknown LITE3_BASE_RPY_SOURCE='" << requested
                          << "'. Falling back to interface_rpy." << std::endl;
                mode = BaseRpySourceMode::kInterfaceRpy;
            }
        }
        base_rpy_source_ = mode;
        base_rpy_source_label_ = BaseRpySourceToString(mode);
        std::cout << "[ONNX] base_rpy source mode: " << base_rpy_source_label_ << std::endl;
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
                              const Vec3f& base_rpy_interface,
                              const Vec3f& base_rpy_from_quat_std,
                              const Vec3f& base_rpy_from_quat_legacy,
                              const Vec3f& projected_gravity,
                              const Vec3f& body_omega,
                              const Vec3f& base_acc,
                              const Vec3f& omega_world,
                              const Vec4f& base_quat_wxyz,
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
        // Auto-resolve training debug directory from deploy build location.
        // Works for both:
        //   /workspace/Lite3_rl_deploy/build  -> /workspace/rl_training_new/lite3_debug/deploy
        //   ~/Lite3/Lite3_rl_deploy/build     -> ~/Lite3/rl_training_new/lite3_debug/deploy
        {
            std::error_code ec;
            std::filesystem::path p = std::filesystem::weakly_canonical(GetAbsPath(), ec);
            if (!ec && !p.empty()) {
                const std::filesystem::path root = p.parent_path().parent_path();  // .../Lite3 or /workspace
                if (!root.empty()) {
                    candidates.emplace_back((root / "rl_training_new" / "lite3_debug" / "deploy").string());
                    candidates.emplace_back((root / "rl_training_new" / "debug_deploy").string());
                }
            }
        }
        if (std::filesystem::exists("/workspace/rl_training_new")) {
            candidates.emplace_back("/workspace/rl_training_new/lite3_debug/deploy");
            candidates.emplace_back("/workspace/rl_training_new/debug_deploy");
        }
        if (const char* home = std::getenv("HOME")) {
            std::filesystem::path h(home);
            candidates.emplace_back((h / "Lite3" / "rl_training_new" / "lite3_debug" / "deploy").string());
            candidates.emplace_back((h / "Lite3" / "rl_training_new" / "debug_deploy").string());
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
        if (run_cnt_ == 0) {
            std::cout << "[DEBUG] deploy dump root: " << dump_root << std::endl;
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
        ofs << "\nbase_rpy_source " << base_rpy_source_label_;
        ofs << "\nbase_rpy_interface";
        for (int i = 0; i < 3; ++i) ofs << " " << base_rpy_interface(i);
        ofs << "\nbase_rpy_from_quat_std";
        for (int i = 0; i < 3; ++i) ofs << " " << base_rpy_from_quat_std(i);
        ofs << "\nbase_rpy_from_quat_legacy";
        for (int i = 0; i < 3; ++i) ofs << " " << base_rpy_from_quat_legacy(i);
        ofs << "\nprojected_gravity";
        for (int i = 0; i < 3; ++i) ofs << " " << projected_gravity(i);
        ofs << "\nbody_omega";
        for (int i = 0; i < 3; ++i) ofs << " " << body_omega(i);
        ofs << "\nbase_acc";
        for (int i = 0; i < 3; ++i) ofs << " " << base_acc(i);
        ofs << "\nomega_world";
        for (int i = 0; i < 3; ++i) ofs << " " << omega_world(i);
        ofs << "\nbase_quat_wxyz";
        for (int i = 0; i < 4; ++i) ofs << " " << base_quat_wxyz(i);
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
        ofs << "\nobs_contract cmd:3 base_rpy:3 body_omega:3 joint_pos:12 joint_vel:12 joint_pos_history:36 joint_vel_history:24 action_history:24";
        ofs << "\nparity_build_id 2026-02-23-base-rpy-source-selector";
        ofs << "\nbase_rpy_mode " << base_rpy_source_label_;
        if (base_rpy_source_ == BaseRpySourceMode::kQuatLegacyXyzw) {
            ofs << "_step0_pi_canonical_neg";
        }
        ofs << "\nhistory_frame_push_mode single_first_step";
        ofs << "\naction_history_mode "
            << (action_hist_delayed_mode_ ? "training_parity_zero_seed_delay1" : (history_seed_mode_ + "_direct"));
        ofs << "\nhistory_seed_file_loaded " << (history_seed_file_loaded_ ? 1 : 0);
        ofs << "\nhistory_seed_file_used " << (history_seed_file_consumed_ ? 1 : 0);
        ofs << "\nhistory_seed_file_path "
            << ((history_seed_file_loaded_ || history_seed_file_consumed_) ? history_seed_file_path_ : "none");
        ofs << "\njoint_pos_history";
        for (const auto& ph : pos_hist_) {
            for (int i = 0; i < kActDim; ++i) ofs << " " << ph(i);
        }
        // legacy alias kept for compatibility with older compare reports
        ofs << "\npos_hist";
        for (const auto& ph : pos_hist_) {
            for (int i = 0; i < kActDim; ++i) ofs << " " << ph(i);
        }
        ofs << "\njoint_vel_history";
        for (const auto& vh : vel_hist_) {
            for (int i = 0; i < kActDim; ++i) ofs << " " << vh(i);
        }
        // legacy alias kept for compatibility with older compare reports
        ofs << "\nvel_hist";
        for (const auto& vh : vel_hist_) {
            for (int i = 0; i < kActDim; ++i) ofs << " " << vh(i);
        }
        ofs << "\naction_history";
        for (const auto& ah : tgt_hist_) {
            for (int i = 0; i < kActDim; ++i) ofs << " " << ah(i);
        }
        // legacy alias kept for compatibility with older compare reports
        ofs << "\ntgt_hist";
        for (const auto& ah : tgt_hist_) {
            for (int i = 0; i < kActDim; ++i) ofs << " " << ah(i);
        }
        ofs << "\naction_raw";
        for (int i = 0; i < kActDim; ++i) ofs << " " << policy_action_raw(i);
        ofs << "\naction_offset";
        for (int i = 0; i < kActDim; ++i) ofs << " " << action_offset(i);
        VecXf target_joint_pos_policy = dof_pos_default_policy_ + action_offset;
        ofs << "\ntarget_joint_pos_policy";
        for (int i = 0; i < kActDim; ++i) ofs << " " << target_joint_pos_policy(i);
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
