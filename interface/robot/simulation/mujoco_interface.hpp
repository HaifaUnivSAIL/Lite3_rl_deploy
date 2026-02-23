/**
 * @file mujoco_interface.hpp
 * @brief simulation in mujoco
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-08-010
 * @copyright Copyright (c) 2025 DeepRobotics
 */


#ifndef MUJOCO_INTERFACE_HPP_
#define MUJOCO_INTERFACE_HPP_

#include "robot_interface.h"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <string>
#include <thread>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <array>


#include <atomic>
#include <mutex>

#include "json.hpp"

namespace interface {

class MujocoInterface : public RobotInterface {
private:
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;

    std::string xml_path_;
    mjvScene scene_;
    mjrContext context_;
    mjvCamera camera_;
    mjvOption opt_;
    GLFWwindow* window_ = nullptr;

    std::thread sim_thread_;

    Vec3d omega_body_, rpy_, acc_;

    VecXd joint_pos_, joint_vel_, joint_tau_;
    VecXd default_joint_pos_;
    Vec4d default_base_quat_;
    Vec3d default_base_pos_;


    // Match training physics step by default (TwoLegStandCfg.sim.dt = 0.005s).
    double dt_ = 0.005;
    double run_time_ = 0.0;
    int run_cnt_ = 0;
    VecXd tau_ff_;
    bool clip_tau_to_training_limits_ = true;
    std::array<double, 12> training_effort_limits_ {{
        24.0, 24.0, 36.0,
        24.0, 24.0, 36.0,
        24.0, 24.0, 36.0,
        24.0, 24.0, 36.0
    }};

    int render_interval_ = 10;
    bool omega_source_world_to_body_ = false;

    std::default_random_engine dre_;
    std::normal_distribution<> gyro_nd_{0.0, 0.0}, rpy_nd_{0.0, 0.0}, acc_nd_{0.0, 0.0};

public:
    MujocoInterface(const std::string& robot_name,
                const std::string& xml_path,
                int dof_num = 12)
        : RobotInterface("MujocoSim", dof_num), xml_path_(xml_path) {

        joint_pos_ = VecXd::Zero(dof_num_);
        joint_tau_ = VecXd::Zero(dof_num_);
        joint_vel_ = VecXd::Zero(dof_num_);
        joint_cmd_ = MatXf::Zero(dof_num_, 5);
        default_joint_pos_.resize(dof_num_);
        default_joint_pos_ <<
            // Match TwoLegStandCfg.init_state.default_joint_angles (training).
            -0.0154048, -0.76697,   1.53761,
             0.0159887, -0.768286,  1.53636,
            -0.0221317, -0.765865,  1.54788,
             0.0224431, -0.767203,  1.54679;
        default_base_pos_ << 0.0, 0.0, 0.32;
        // MuJoCo quaternion order: (w, x, y, z). Training uses (x, y, z, w).
        default_base_quat_ << 0.9999929146412841, -0.00023085526184233324, -0.0032073138974974646, -0.0019571690372445424;

        

        std::cout << "[MuJoCoInterface] Loading model: " << xml_path_ << std::endl;
        char error[1000] = "";
        model_ = mj_loadXML(xml_path_.c_str(), 0, error, 1000);
        if (!model_) {
            std::cerr << "[ERROR] Failed to load MuJoCo model: " << error << std::endl;
            exit(1);
        }
        data_ = mj_makeData(model_);

        // Allow overriding simulation timestep for deploy parity (training uses 0.005s).
        if (const char* dt_env = std::getenv("LITE3_MUJOCO_DT")) {
            char* endptr = nullptr;
            const double parsed = std::strtod(dt_env, &endptr);
            if (endptr != dt_env && std::isfinite(parsed) && parsed > 0.0) {
                dt_ = parsed;
            }
        }
        // Optional render interval override.
        if (const char* render_env = std::getenv("LITE3_MUJOCO_RENDER_INTERVAL")) {
            const int parsed = std::atoi(render_env);
            if (parsed > 0) {
                render_interval_ = parsed;
            }
        }
        auto parse_nonnegative_std = [](const char* env_key, double fallback) -> double {
            const char* env = std::getenv(env_key);
            if (!env) return fallback;
            char* endptr = nullptr;
            const double parsed = std::strtod(env, &endptr);
            if (endptr == env || !std::isfinite(parsed) || parsed < 0.0) {
                return fallback;
            }
            return parsed;
        };
        const double gyro_noise_std = parse_nonnegative_std("LITE3_IMU_GYRO_NOISE_STD", 0.0);
        const double rpy_noise_std = parse_nonnegative_std("LITE3_IMU_RPY_NOISE_STD", 0.0);
        const double acc_noise_std = parse_nonnegative_std("LITE3_IMU_ACC_NOISE_STD", 0.0);
        if (const char* clip_env = std::getenv("LITE3_TAU_CLIP_TRAINING_LIMITS")) {
            clip_tau_to_training_limits_ = std::atoi(clip_env) != 0;
        }
        // Keep hardware-aligned convention by default:
        // qvel[3:6] is forwarded as body omega unless explicitly overridden.
        std::string omega_source = "qvel_body";
        if (const char* omega_env = std::getenv("LITE3_MUJOCO_OMEGA_SOURCE")) {
            omega_source = omega_env;
            std::transform(omega_source.begin(), omega_source.end(), omega_source.begin(),
                           [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        }
        if (omega_source == "world_to_body") {
            omega_source_world_to_body_ = true;
        } else if (omega_source != "qvel_body") {
            std::cerr << "[MuJoCoInterface][WARN] Unknown LITE3_MUJOCO_OMEGA_SOURCE='"
                      << omega_source << "', fallback to qvel_body\n";
            omega_source_world_to_body_ = false;
        }
        gyro_nd_ = std::normal_distribution<>(0.0, gyro_noise_std);
        rpy_nd_ = std::normal_distribution<>(0.0, rpy_noise_std);
        acc_nd_ = std::normal_distribution<>(0.0, acc_noise_std);
        model_->opt.timestep = dt_;
        std::cout << "[MuJoCoInterface] dt=" << dt_ << "s, render_interval=" << render_interval_ << std::endl;
        std::cout << "[MuJoCoInterface] imu_noise_std rpy=" << rpy_noise_std
                  << " gyro=" << gyro_noise_std
                  << " acc=" << acc_noise_std << std::endl;
        std::cout << "[MuJoCoInterface] omega_source="
                  << (omega_source_world_to_body_ ? "world_to_body" : "qvel_body") << std::endl;
        std::cout << "[MuJoCoInterface] tau_clip_training_limits="
                  << (clip_tau_to_training_limits_ ? "on" : "off") << std::endl;

        // 可视化初始化
        // mjv_defaultCamera(&camera_);
        // mjv_defaultOption(&opt_);
        // mjv_defaultScene(&scene_);
        // mjr_defaultContext(&context_);

        // if (!glfwInit()) {
        //     std::cerr << "[ERROR] Could not initialize GLFW" << std::endl;
        //     exit(1);
        // }

        // window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulation", NULL, NULL);
        // if (!window_) {
        //     std::cerr << "[ERROR] Could not create GLFW window" << std::endl;
        //     glfwTerminate();
        //     exit(1);
        // }

        // glfwMakeContextCurrent(window_);
        // mjv_makeScene(model_, &scene_, 2000);
        // mjr_makeContext(model_, &context_, mjFONTSCALE_150);

        std::cout << "[MuJoCoInterface] Model loaded successfully. DOF: " << model_->nu << std::endl;

        camera_.type = mjCAMERA_TRACKING;
        camera_.trackbodyid = mj_name2id(model_, mjOBJ_BODY, "TORSO");  // “base”为MJCF中主体名
        // camera_.lookat[0] = 0.0;
        // camera_.lookat[1] = 0.0;
        // camera_.lookat[2] = 1.0;
        camera_.distance = 4.0;
        camera_.azimuth = 90.0;
        camera_.elevation = -30.0;
    }

    ~MujocoInterface() {
        mj_deleteData(data_);
        mj_deleteModel(model_);
        mjv_freeScene(&scene_);
        mjr_freeContext(&context_);
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

    virtual double GetInterfaceTimeStamp() override { return run_time_; }

    // virtual VecXf GetJointPosition() override { return joint_pos_; }
    // virtual VecXf GetJointVelocity() override { return joint_vel_; }
    // virtual VecXf GetJointTorque() override { return joint_tau_; }
    // virtual Vec3f GetImuRpy() override { return rpy_; }
    // virtual Vec3f GetImuAcc() override { return acc_; }
    // virtual Vec3f GetImuOmega() override { return omega_body_; }
    // virtual VecXf GetContactForce() override { return VecXf::Zero(4); }

    virtual VecXf GetJointPosition() override { return joint_pos_.cast<float>(); }
    virtual VecXf GetJointVelocity() override { return joint_vel_.cast<float>(); }
    virtual VecXf GetJointTorque() override { return joint_tau_.cast<float>(); }
    virtual Vec3f GetImuRpy() override { return rpy_.cast<float>(); }
    virtual Vec3f GetImuAcc() override { return acc_.cast<float>(); }
    virtual Vec3f GetImuOmega() override { return omega_body_.cast<float>(); }
    virtual Vec4f GetImuQuat() override {
        // MuJoCo stores free-joint quaternion in qpos[3:7] as (w, x, y, z).
        return Eigen::Map<Vec4d>(data_->qpos + 3, 4).cast<float>();
    }
    virtual VecXf GetContactForce() override { return VecXf::Zero(4); }



    virtual void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5> input) override {
        joint_cmd_ = input;
    }

    virtual void Start() override {
        start_flag_ = true;
        sim_thread_ = std::thread(std::bind(&MujocoInterface::Run, this));
    }

    virtual void Stop() override {
        start_flag_ = false;
        sim_thread_.join();
    }

    double GetSimulationDt() const {
        return dt_;
    }

private:
    void Run() {


        // 可视化初始化
        // mjv_defaultCamera(&camera_);
        mjv_defaultOption(&opt_);
        mjv_defaultScene(&scene_);
        mjr_defaultContext(&context_);

        if (!glfwInit()) {
            std::cerr << "[ERROR] Could not initialize GLFW" << std::endl;
            exit(1);
        }

        window_ = glfwCreateWindow(1200, 900, "MuJoCo Simulation", NULL, NULL);
        if (!window_) {
            std::cerr << "[ERROR] Could not create GLFW window" << std::endl;
            glfwTerminate();
            exit(1);
        }

        glfwMakeContextCurrent(window_);
        mjv_makeScene(model_, &scene_, 2000);
        mjr_makeContext(model_, &context_, mjFONTSCALE_150);

        glfwMakeContextCurrent(window_);

        glfwSwapInterval(1);

        ApplyInitialPose();
        
        
        
        while (start_flag_ && !glfwWindowShouldClose(window_))  {
            run_time_ = run_cnt_ * dt_;

            UpdateImu();
            UpdateJointState();
            ApplyControl();

            mj_step(model_, data_);

            if (run_cnt_ % render_interval_ == 0) {
                Render();
            }        
            // std::cout << "Rendered frame " << run_cnt_ << std::endl;

            ++run_cnt_;
            std::this_thread::sleep_for(std::chrono::microseconds(int(dt_ * 1e6)));
        }
    }

    void UpdateImu() {

        double* q = data_->qpos + 3;
        double qw = q[0], qx = q[1], qy = q[2], qz = q[3];

        double roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
        double pitch = asin(2 * (qw * qy - qz * qx));
        double yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));

        rpy_ << roll + rpy_nd_(dre_), pitch + rpy_nd_(dre_), yaw + rpy_nd_(dre_);

        acc_ = Eigen::Map<Vec3d>(data_->sensordata + 16);

        Vec3d omega_body = Eigen::Map<Vec3d>(data_->qvel + 3);
        if (omega_source_world_to_body_) {
            // Debug-only alternative path for convention checks.
            Vec3d omega_world = omega_body;
            Eigen::Quaterniond quat_world_from_body(qw, qx, qy, qz);
            quat_world_from_body.normalize();
            omega_body = quat_world_from_body.conjugate() * omega_world;
        }
        omega_body_ = omega_body + Vec3d(gyro_nd_(dre_), gyro_nd_(dre_), gyro_nd_(dre_));

        // std::cout << "[IMU] RPY: " << rpy_.transpose()
        //       << " | Omega: " << omega_body_.transpose()
        //       << " | Acc: " << acc_.transpose() << std::endl;
        
        
    }

    void UpdateJointState() {
        joint_pos_ = Eigen::Map<VecXd>(data_->qpos + 7, dof_num_);
        joint_vel_ = Eigen::Map<VecXd>(data_->qvel + 6, dof_num_);
        joint_tau_ = Eigen::Map<VecXd>(data_->ctrl, dof_num_);

        // std::cout << "[JointState] pos[0:3]: " << joint_pos_.head(3).transpose()
        //       << " | vel[0:3]: " << joint_vel_.head(3).transpose()
        //       << " | tau[0:3]: " << joint_tau_.head(3).transpose() << std::endl;
        
    }

    void ApplyControl() {
        auto kp = joint_cmd_.col(0).cast<double>();
        auto q_des = joint_cmd_.col(1).cast<double>();
        auto kd = joint_cmd_.col(2).cast<double>();
        auto dq_des = joint_cmd_.col(3).cast<double>();
        auto tau_ff = joint_cmd_.col(4).cast<double>();

        VecXd tau_out = kp.cwiseProduct(q_des - joint_pos_)
                      + kd.cwiseProduct(dq_des - joint_vel_)
                      + tau_ff;
        if (clip_tau_to_training_limits_ && dof_num_ == 12) {
            for (int i = 0; i < dof_num_; ++i) {
                const double lim = training_effort_limits_[i];
                tau_out(i) = std::min(std::max(tau_out(i), -lim), lim);
            }
        }
        
        // std::cout << "[ApplyCtrl] tau_out[0:3]: " << tau_out.head(3).transpose()
        //       << " | q_des[0:3]: " << q_des.head(3).transpose()
        //       << " | q[0:3]: " << joint_pos_.head(3).transpose() << std::endl;                  

        // VecXd tau_out = joint_cmd_.col(0).cwiseProduct(joint_cmd_.col(1) - joint_pos_)
        //               + joint_cmd_.col(2).cwiseProduct(joint_cmd_.col(3) - joint_vel_)
        //               + joint_cmd_.col(4);
        Eigen::Map<VecXd>(data_->ctrl, dof_num_) = tau_out;
    }

    void Render() {
        mjv_updateScene(model_, data_, &opt_, nullptr, &camera_, mjCAT_ALL, &scene_);
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
        mjr_render(viewport, &scene_, &context_);
        glfwSwapBuffers(window_);
        glfwPollEvents();
    }

    void ApplyInitialPose() {
        mj_resetData(model_, data_);

        // If provided, prefer a deterministic training snapshot (used to reproduce obs/action diffs).
        // JSON format matches `Lite3_rl_training/legged_gym/legged_gym/envs/base/deploy_snapshot.json`.
        const char* deploy_state = std::getenv("LITE3_DEPLOY_STATE");
        if (deploy_state && std::filesystem::exists(deploy_state)) {
            try {
                std::ifstream ifs(deploy_state);
                nlohmann::json j;
                ifs >> j;

                auto read_vec = [&](const char* key, int n) -> std::vector<double> {
                    if (!j.contains(key) || !j[key].is_array() || static_cast<int>(j[key].size()) != n) {
                        return {};
                    }
                    std::vector<double> out;
                    out.reserve(n);
                    for (int i = 0; i < n; ++i) out.push_back(j[key][i].get<double>());
                    return out;
                };

                const auto base_pos = read_vec("base_pos", 3);
                const auto base_quat_xyzw = read_vec("base_quat", 4);
                const auto base_lin_vel = read_vec("base_lin_vel", 3);
                const auto base_ang_vel = read_vec("base_ang_vel", 3);
                const auto joint_pos = read_vec("joint_pos", dof_num_);
                const auto joint_vel = read_vec("joint_vel", dof_num_);

                if (base_pos.size() == 3) {
                    data_->qpos[0] = base_pos[0];
                    data_->qpos[1] = base_pos[1];
                    data_->qpos[2] = base_pos[2];
                }
                if (base_quat_xyzw.size() == 4) {
                    // Convert XYZW -> WXYZ for MuJoCo.
                    data_->qpos[3] = base_quat_xyzw[3];
                    data_->qpos[4] = base_quat_xyzw[0];
                    data_->qpos[5] = base_quat_xyzw[1];
                    data_->qpos[6] = base_quat_xyzw[2];
                } else {
                    data_->qpos[3] = default_base_quat_(0);
                    data_->qpos[4] = default_base_quat_(1);
                    data_->qpos[5] = default_base_quat_(2);
                    data_->qpos[6] = default_base_quat_(3);
                }

                if (joint_pos.size() == static_cast<size_t>(dof_num_)) {
                    for (int i = 0; i < dof_num_; ++i) {
                        data_->qpos[7 + i] = joint_pos[i];
                    }
                } else {
                    for (int i = 0; i < dof_num_; ++i) {
                        data_->qpos[7 + i] = default_joint_pos_(i);
                    }
                }

                std::fill_n(data_->qvel, model_->nv, 0.0);
                if (base_lin_vel.size() == 3) {
                    data_->qvel[0] = base_lin_vel[0];
                    data_->qvel[1] = base_lin_vel[1];
                    data_->qvel[2] = base_lin_vel[2];
                }
                if (base_ang_vel.size() == 3) {
                    data_->qvel[3] = base_ang_vel[0];
                    data_->qvel[4] = base_ang_vel[1];
                    data_->qvel[5] = base_ang_vel[2];
                }
                if (joint_vel.size() == static_cast<size_t>(dof_num_)) {
                    for (int i = 0; i < dof_num_; ++i) {
                        data_->qvel[6 + i] = joint_vel[i];
                    }
                }

                std::fill_n(data_->ctrl, model_->nu, 0.0);
                mj_forward(model_, data_);
                UpdateImu();
                UpdateJointState();
                std::cout << "[MUJOCO RESET] Loaded LITE3_DEPLOY_STATE=" << deploy_state << "\n";
                run_time_ = 0.0;
                run_cnt_ = 0;
                return;
            } catch (const std::exception& e) {
                std::cerr << "[MUJOCO RESET] Failed to parse LITE3_DEPLOY_STATE (" << deploy_state
                          << "): " << e.what() << "\n";
            }
        }

        // Default to deterministic reset for deploy parity experiments.
        bool use_random_reset = false;
        if (const char* env = std::getenv("LITE3_RANDOM_RESET")) {
            // Set LITE3_RANDOM_RESET=1 to enable randomization.
            use_random_reset = std::atoi(env) != 0;
        }
        Eigen::VectorXd init_joint = default_joint_pos_;
        if (use_random_reset) {
            std::uniform_real_distribution<double> dist(0.5, 1.5);
            for (int i = 0; i < init_joint.size(); ++i) {
                init_joint(i) *= dist(dre_);
            }
        }

        data_->qpos[0] = default_base_pos_(0);
        data_->qpos[1] = default_base_pos_(1);
        data_->qpos[2] = default_base_pos_(2);
        data_->qpos[3] = default_base_quat_(0);
        data_->qpos[4] = default_base_quat_(1);
        data_->qpos[5] = default_base_quat_(2);
        data_->qpos[6] = default_base_quat_(3);

        for (int i = 0; i < dof_num_; ++i) {
            data_->qpos[7 + i] = init_joint(i);
        }

        std::fill_n(data_->qvel, model_->nv, 0.0);
        if (use_random_reset) {
            // Match training reset: base velocities uniform(-0.5, 0.5).
            std::uniform_real_distribution<double> vdist(-0.5, 0.5);
            for (int i = 0; i < 6 && i < model_->nv; ++i) {
                data_->qvel[i] = vdist(dre_);
            }
        }
        std::fill_n(data_->ctrl, model_->nu, 0.0);

        mj_forward(model_, data_);
        UpdateImu();
        UpdateJointState();
        std::cout << "[MUJOCO RESET] base_pos "
                  << default_base_pos_.transpose()
                  << " base_quat "
                  << default_base_quat_.transpose()
                  << "\n[MUJOCO RESET] joint_pos " << joint_pos_.transpose()
                  << "\n";
        run_time_ = 0.0;
        run_cnt_ = 0;
    }

public:
    void PrintFullState() {
        std::cout << "[MUJOCO STATE] base_pos " << Eigen::Map<Vec3d>(data_->qpos, 3).transpose()
                  << "\n[MUJOCO STATE] base_quat "
                  << Eigen::Map<Vec4d>(data_->qpos + 3, 4).transpose()
                  << "\n[MUJOCO STATE] base_lin_vel "
                  << Eigen::Map<Vec3d>(data_->qvel, 3).transpose()
                  << "\n[MUJOCO STATE] base_ang_vel "
                  << Eigen::Map<Vec3d>(data_->qvel + 3, 3).transpose()
                  << "\n[MUJOCO STATE] joint_pos "
                  << Eigen::Map<VecXd>(data_->qpos + 7, dof_num_).transpose()
                  << "\n[MUJOCO STATE] joint_vel "
                  << Eigen::Map<VecXd>(data_->qvel + 6, dof_num_).transpose()
                  << "\n";
    }
};

}  // namespace interface

#endif  // MUJOCO_INTERFACE_HPP_
