/**
 * @file rl_control_state_onnx.hpp
 * @brief rl policy runnning state using onnx
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-08-10
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */




#ifndef RL_CONTROL_STATE_ONNX_HPP_
#define RL_CONTROL_STATE_ONNX_HPP_

#include "state_base.h"
#include "policy_runner_base.hpp"
#include "lite3_test_policy_runner_onnx.hpp"
#if defined(BUILD_SIMULATION) && defined(USE_MJCPP)
    #include "../interface/robot/simulation/mujoco_interface.hpp"
#endif
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>



class RLControlStateONNX : public StateBase
{
private:
    RobotBasicState rbs_;
    std::atomic<int> state_run_cnt_{-1};
    mutable std::mutex rbs_mutex_;
    Vec3f prev_base_rpy_ = Vec3f::Zero();
    bool has_prev_orientation_ = false;
    double prev_timestamp_sec_ = 0.0;
    Vec3f fixed_cmd_{0.f, 0.f, 0.f}; // training two-leg stand uses zero commands by default
    bool fixed_cmd_enabled_ = false;
    std::string last_fixed_cmd_env_;
    bool printed_initial_obs_ = false;

    std::shared_ptr<PolicyRunnerBase> policy_ptr_;
    std::shared_ptr<Lite3TestPolicyRunnerONNX> test_policy_;


    
    std::thread run_policy_thread_;
    std::atomic<bool> start_flag_{true};
    bool async_policy_mode_ = false;
    int policy_step_record_ = -1;

    float policy_cost_time_ = 1;

    void UpdateRobotObservation(){
        const double current_ts = ri_ptr_->GetInterfaceTimeStamp();
        RobotBasicState next_rbs;
        next_rbs.base_rpy     = ri_ptr_->GetImuRpy();
        next_rbs.base_quat    = ri_ptr_->GetImuQuat();
        const Vec4f quat_identity(1.f, 0.f, 0.f, 0.f);
        const bool quat_is_placeholder_identity =
            next_rbs.base_quat.allFinite() &&
            (next_rbs.base_quat - quat_identity).cwiseAbs().maxCoeff() < 1e-6f;
        if (!next_rbs.base_quat.allFinite() ||
            next_rbs.base_quat.norm() < 1e-6f ||
            quat_is_placeholder_identity) {
            // Fallback when interface does not provide a valid quaternion
            // (or returns the base-class identity placeholder): rebuild from IMU rpy.
            Mat3f R = RpyToRm(next_rbs.base_rpy);
            Eigen::Quaternionf q(R);
            q.normalize();
            next_rbs.base_quat << q.w(), q.x(), q.y(), q.z();
        } else {
            next_rbs.base_quat.normalize();
        }
        Eigen::Quaternionf quat_wxyz(next_rbs.base_quat(0), next_rbs.base_quat(1),
                                     next_rbs.base_quat(2), next_rbs.base_quat(3));
        next_rbs.base_rot_mat = quat_wxyz.normalized().toRotationMatrix();
        next_rbs.projected_gravity = RmToProjectedGravity(next_rbs.base_rot_mat);
        Vec3f imu_omega   = ri_ptr_->GetImuOmega();
        if (has_prev_orientation_) {
            const double dt = current_ts - prev_timestamp_sec_;
            const float omega_max = imu_omega.cwiseAbs().maxCoeff();
            if (omega_max < 1e-4f && dt > 1e-6) {
                Mat3f R_prev = RpyToRm(prev_base_rpy_);
                Mat3f delta  = R_prev.transpose() * next_rbs.base_rot_mat;
                Eigen::AngleAxisf aa(delta);
                float angle = aa.angle();
                if (!std::isnan(angle) && std::abs(angle) > 1e-8f) {
                    Vec3f axis = aa.axis();
                    imu_omega = axis * (angle / static_cast<float>(dt));
                }
            }
        }
        next_rbs.base_omega   = imu_omega;
        next_rbs.base_acc     = ri_ptr_->GetImuAcc();
        next_rbs.joint_pos    = ri_ptr_->GetJointPosition();
        next_rbs.joint_vel    = ri_ptr_->GetJointVelocity();
        next_rbs.joint_tau    = ri_ptr_->GetJointTorque();
        // static Vec3f cmd_vel;
        // Vec3f cmd_vel_input = Vec3f(uc_ptr_->GetUserCommand().forward_vel_scale, 
        //                             uc_ptr_->GetUserCommand().side_vel_scale, 
        //                             uc_ptr_->GetUserCommand().turnning_vel_scale);

        // Eigen::Vector3f vel_delta = cmd_vel_input - cmd_vel;
        // const Eigen::Vector3f vel_delta_const(0.0015, 0.001, 0.0012);
        // for(int i=0;i<3;++i){
        //     if(fabs(vel_delta(i)) > vel_delta_const(i)) vel_delta(i) = Sign(vel_delta(i))*vel_delta_const(i);
        // }
        // cmd_vel+=vel_delta;           
        // rbs_.cmd_vel_normlized = cmd_vel;
        ParseFixedCommandEnv();
        if (fixed_cmd_enabled_) {
            next_rbs.cmd_vel_normlized = fixed_cmd_;
        } else {
            next_rbs.cmd_vel_normlized = Vec3f(uc_ptr_->GetUserCommand().forward_vel_scale, 
                                        uc_ptr_->GetUserCommand().side_vel_scale, 
                                        uc_ptr_->GetUserCommand().turnning_vel_scale);
        }
        prev_base_rpy_ = next_rbs.base_rpy;
        prev_timestamp_sec_ = current_ts;
        has_prev_orientation_ = true;

        {
            std::lock_guard<std::mutex> lock(rbs_mutex_);
            rbs_ = next_rbs;
        }

        if (!printed_initial_obs_) {
            std::cout << "[RLControlStateONNX] Initial obs snapshot\n"
                      << "  cmd: " << next_rbs.cmd_vel_normlized.transpose() << "\n"
                      << "  base_rpy: " << next_rbs.base_rpy.transpose() << "\n"
                      << "  joint_pos: " << next_rbs.joint_pos.transpose() << "\n"
                      << "  joint_vel: " << next_rbs.joint_vel.transpose() << "\n";
            printed_initial_obs_ = true;
        }
        
    }

    void ExecutePolicyIfDue(int step_idx) {
        if (step_idx < 0) return;
        if (step_idx % policy_ptr_->decimation_ != 0) return;
        if (step_idx == policy_step_record_) return;
        RobotBasicState rbs_snapshot;
        {
            std::lock_guard<std::mutex> lock(rbs_mutex_);
            rbs_snapshot = rbs_;
        }
        timespec start_timestamp, end_timestamp;
        clock_gettime(CLOCK_MONOTONIC, &start_timestamp);
        auto ra = policy_ptr_->GetRobotAction(rbs_snapshot);
        MatXf res = ra.ConvertToMat();
        ri_ptr_->SetJointCommand(res);
        policy_step_record_ = step_idx;
        clock_gettime(CLOCK_MONOTONIC, &end_timestamp);
        policy_cost_time_ = (end_timestamp.tv_sec - start_timestamp.tv_sec) * 1e3
                            + (end_timestamp.tv_nsec - start_timestamp.tv_nsec) / 1e6;
    }

    void PolicyRunner(){
        while (start_flag_.load(std::memory_order_acquire)){
            const int step_idx = state_run_cnt_.load(std::memory_order_acquire);
            ExecutePolicyIfDue(step_idx);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void ConfigurePolicyTimingFromSim() {
#if defined(BUILD_SIMULATION) && defined(USE_MJCPP)
        auto mujoco_if = std::dynamic_pointer_cast<interface::MujocoInterface>(ri_ptr_);
        if (!mujoco_if) {
            return;
        }

        const double sim_dt = mujoco_if->GetSimulationDt();
        if (!(std::isfinite(sim_dt) && sim_dt > 0.0)) {
            return;
        }

        // Match training control period by default: sim.dt=0.005, decimation=4 -> 0.02s.
        double target_ctrl_dt = 0.02;
        if (const char* env_ctrl_dt = std::getenv("LITE3_POLICY_CONTROL_DT")) {
            char* endptr = nullptr;
            const double parsed = std::strtod(env_ctrl_dt, &endptr);
            if (endptr != env_ctrl_dt && std::isfinite(parsed) && parsed > 0.0) {
                target_ctrl_dt = parsed;
            }
        }

        int decimation = std::max(1, static_cast<int>(std::lround(target_ctrl_dt / sim_dt)));
        if (const char* dec_env = std::getenv("LITE3_POLICY_DECIMATION")) {
            const int parsed = std::atoi(dec_env);
            if (parsed > 0) {
                decimation = parsed;
            }
        }

        policy_ptr_->SetDecimation(decimation);
        const double applied_ctrl_dt = sim_dt * static_cast<double>(decimation);
        std::cout << "[RLControlStateONNX] sim_dt=" << sim_dt
                  << ", policy_decimation=" << decimation
                  << ", policy_control_dt=" << applied_ctrl_dt << "s" << std::endl;
#endif
    }

public:
    RLControlStateONNX(const RobotType& robot_type, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_type, state_name, data_ptr){
        rbs_.base_rpy.setZero();
        rbs_.projected_gravity.setZero();
        rbs_.base_quat << 1.f, 0.f, 0.f, 0.f;
        rbs_.base_rot_mat.setIdentity();
        rbs_.base_omega.setZero();
        rbs_.base_acc.setZero();
        rbs_.cmd_vel_normlized.setZero();
        rbs_.joint_pos = VecXf::Zero(12);
        rbs_.joint_vel = VecXf::Zero(12);
        rbs_.joint_tau = VecXf::Zero(12);
        test_policy_ = std::make_shared<Lite3TestPolicyRunnerONNX>("test_onnx");
        policy_ptr_ = test_policy_;
        if(!policy_ptr_){
            std::cerr << "[ERROR] Failed to initialize ONNX policy runner." << std::endl;
            exit(0);
        }  
        policy_ptr_->DisplayPolicyInfo();
        }
    ~RLControlStateONNX(){}

    virtual void OnEnter() {
        state_run_cnt_.store(-1, std::memory_order_release);
        start_flag_.store(true, std::memory_order_release);
        policy_step_record_ = -1;
        ConfigurePolicyTimingFromSim();
        policy_ptr_->OnEnter();
        async_policy_mode_ = false;
        if (const char* async_env = std::getenv("LITE3_POLICY_ASYNC")) {
            async_policy_mode_ = std::atoi(async_env) != 0;
        }
        std::cout << "[RLControlStateONNX] policy_execution_mode="
                  << (async_policy_mode_ ? "async_thread" : "sync_main_loop") << std::endl;
        if (async_policy_mode_) {
            run_policy_thread_ = std::thread(std::bind(&RLControlStateONNX::PolicyRunner, this));
        }
#if defined(BUILD_SIMULATION) && defined(USE_MJCPP)
        if (auto mujoco_if = std::dynamic_pointer_cast<interface::MujocoInterface>(ri_ptr_)) {
            mujoco_if->PrintFullState();
        }
#endif
        StateBase::msfb_.UpdateCurrentState(RobotMotionState::RLControlMode);
        uc_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };

    virtual void OnExit() { 
        start_flag_.store(false, std::memory_order_release);
        if (run_policy_thread_.joinable()) {
            run_policy_thread_.join();
        }
        state_run_cnt_.store(-1, std::memory_order_release);
    }

    virtual void Run() {
        UpdateRobotObservation();
        const int step_idx = state_run_cnt_.fetch_add(1, std::memory_order_release) + 1;
        if (!async_policy_mode_) {
            ExecutePolicyIfDue(step_idx);
        }
        ds_ptr_->InsertScopeData(0, policy_cost_time_);
    }

    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::JointDamping)) return true;
        return PostureUnsafeCheck();
    }

    bool PostureUnsafeCheck(){
        const char* disable = std::getenv("LITE3_DISABLE_POSTURE_CHECK");
        if (disable && std::atoi(disable) != 0) {
            return false;
        }
        auto readLimit = [](const char* env_name, float default_deg) -> float {
            const char* env_val = std::getenv(env_name);
            if (!env_val) {
                return default_deg;
            }
            char* endptr = nullptr;
            float parsed = std::strtof(env_val, &endptr);
            if (endptr == env_val || !std::isfinite(parsed) || parsed <= 0.f) {
                return default_deg;
            }
            return parsed;
        };
        // Defaults tuned for the two-leg stand task (torso can intentionally reach large pitch).
        const float roll_limit_deg  = readLimit("LITE3_POSTURE_LIMIT_ROLL_DEG", 40.f);
        const float pitch_limit_deg = readLimit("LITE3_POSTURE_LIMIT_PITCH_DEG", 90.f);
        constexpr float safety_margin_deg = 0.5f;
        const float roll_limit_rad  = (roll_limit_deg  + safety_margin_deg) * static_cast<float>(M_PI) / 180.f;
        const float pitch_limit_rad = (pitch_limit_deg + safety_margin_deg) * static_cast<float>(M_PI) / 180.f;
        Vec3f rpy = ri_ptr_->GetImuRpy();
        if(fabs(rpy(0)) > roll_limit_rad || fabs(rpy(1)) > pitch_limit_rad){
            std::cout << "[RLControlStateONNX] posture unsafe (deg): " << 180./M_PI*rpy.transpose()
                      << " | limits (deg): roll " << roll_limit_deg << " pitch " << pitch_limit_deg
                      << " | set LITE3_POSTURE_LIMIT_* or LITE3_DISABLE_POSTURE_CHECK=1\n";
            return true;
        }
        return false;
    }

    void ParseFixedCommandEnv() {
        const char* env = std::getenv("LITE3_FIXED_CMD");
        if (!env) return;
        // Avoid spamming stdout when running in a tight loop.
        if (last_fixed_cmd_env_ == env) return;
        last_fixed_cmd_env_ = env;
        // Special values to disable the fixed command
        if (std::strcmp(env, "disable") == 0 || std::strcmp(env, "none") == 0) {
            fixed_cmd_enabled_ = false;
            std::cout << "[RLControlStateONNX] Fixed command disabled; using user commands.\n";
            return;
        }
        float x=0.f,y=0.f,z=0.f;
        if (std::sscanf(env, "%f %f %f", &x, &y, &z) == 3) {
            fixed_cmd_ << x, y, z;
            fixed_cmd_enabled_ = true;
            std::cout << "[RLControlStateONNX] Fixed command override "
                      << "(" << fixed_cmd_(0) << ", " << fixed_cmd_(1) << ", " << fixed_cmd_(2) << ")\n";
        }
    }

    virtual StateName GetNextStateName() {
        return StateName::kRLControl;
    }
};


#endif  // RL_CONTROL_STATE_ONNX_HPP_
