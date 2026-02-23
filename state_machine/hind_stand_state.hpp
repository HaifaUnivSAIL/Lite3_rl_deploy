/**
 * @file hind_stand_state.hpp
 * @brief from any current posture to a hind-leg standing posture
 */
#ifndef HIND_STAND_STATE_HPP_
#define HIND_STAND_STATE_HPP_

#include "state_base.h"
#include <algorithm>
#include <sstream>

class HindStandState : public StateBase{
private:
    VecXf init_joint_pos_, init_joint_vel_, current_joint_pos_, current_joint_vel_;
    float time_stamp_record_ = 0.f, run_time_ = 0.f;
    VecXf goal_joint_pos_, kp_, kd_;
    MatXf joint_cmd_;
    float transition_duration_ = 2.f;
    VecXf joint_pos_lower_, joint_pos_upper_;
    float torso_pitch_target_ = 1.13f;         // ~65 deg, rear-up / sit-up target
    float pitch_kp_hipy_ = 0.22f;              // rad joint / rad pitch error
    float pitch_kp_knee_ = 0.30f;              // rad joint / rad pitch error
    float roll_kp_hind_ = 0.12f;               // rad joint / rad roll error
    float front_tuck_gain_ = 0.20f;            // front tuck coupling to hind regulation
    float imu_pitch_sign_ = 1.f;               // +1/-1 to align IMU pitch convention
    float front_hipy_min_tuck_ = 0.24f;        // keep front legs detached
    float front_knee_min_tuck_ = 2.45f;        // keep front legs detached
    bool posture_feedback_enable_ = false;
    bool allow_env_joint_override_ = false;
    bool pitch_auto_sign_flip_ = true;
    float pitch_gain_sign_ = 1.f;
    float enter_pitch_err_abs_ = 1e6f;
    float prev_pitch_err_abs_ = 1e6f;
    int pitch_worsen_count_ = 0;
    bool one_shot_sign_check_done_ = false;
    float last_debug_print_time_ = -1.f;

    void GetRobotJointValue(){
        current_joint_pos_ = ri_ptr_->GetJointPosition();
        current_joint_vel_ = ri_ptr_->GetJointVelocity();
        run_time_ = ri_ptr_->GetInterfaceTimeStamp();
    }

    void RecordJointData(){
        init_joint_pos_ = current_joint_pos_;
        init_joint_vel_ = current_joint_vel_;
        time_stamp_record_ = run_time_;
    }

    float GetCubicSplinePos(float x0, float v0, float xf, float vf, float t, float T){
        if(t >= T) return xf;
        float a, b, c, d;
        d = x0;
        c = v0;
        a = (vf*T - 2*xf + v0*T + 2*x0) / pow(T, 3);
        b = (3*xf - vf*T - 2*v0*T - 3*x0) / pow(T, 2);
        return a*pow(t, 3)+b*pow(t, 2)+c*t+d;
    }

    float GetCubicSplineVel(float x0, float v0, float xf, float vf, float t, float T){
        if(t >= T) return 0.f;
        float a, b, c;
        c = v0;
        a = (vf*T - 2*xf + v0*T + 2*x0) / pow(T, 3);
        b = (3*xf - vf*T - 2*v0*T - 3*x0) / pow(T, 2);
        return 3.f*a*pow(t, 2) + 2.f*b*t + c;
    }

    bool TryLoadJointTargetFromEnv(){
        if (!allow_env_joint_override_) {
            if (std::getenv("LITE3_HIND_STAND_JOINT_POS")) {
                std::cout << "[HindStandState] Ignoring LITE3_HIND_STAND_JOINT_POS (set LITE3_HIND_STAND_ALLOW_JOINT_OVERRIDE=1 to enable)\n";
            }
            return false;
        }
        const char* env = std::getenv("LITE3_HIND_STAND_JOINT_POS");
        if (!env) return false;
        std::stringstream ss(env);
        std::vector<float> values;
        float v = 0.f;
        while (ss >> v) {
            values.push_back(v);
            if (ss.peek() == ',' || ss.peek() == ';') ss.ignore();
        }
        if (values.size() != 12) {
            std::cerr << "[HindStandState] Ignore LITE3_HIND_STAND_JOINT_POS: expected 12 floats, got "
                      << values.size() << std::endl;
            return false;
        }
        for (int i = 0; i < 12; ++i) goal_joint_pos_(i) = values[i];
        std::cout << "[HindStandState] Loaded joint target from LITE3_HIND_STAND_JOINT_POS\n";
        return true;
    }

    void MaybeLoadPostureFeedbackFromEnv(){
        auto parseEnv = [](const char* key, float fallback) -> float {
            const char* env = std::getenv(key);
            if (!env) return fallback;
            char* endptr = nullptr;
            float parsed = std::strtof(env, &endptr);
            if (endptr == env || !std::isfinite(parsed)) return fallback;
            return parsed;
        };
        torso_pitch_target_ = parseEnv("LITE3_HIND_STAND_PITCH_TARGET", torso_pitch_target_);
        pitch_kp_hipy_ = parseEnv("LITE3_HIND_STAND_PITCH_KP_HIPY", pitch_kp_hipy_);
        pitch_kp_knee_ = parseEnv("LITE3_HIND_STAND_PITCH_KP_KNEE", pitch_kp_knee_);
        roll_kp_hind_ = parseEnv("LITE3_HIND_STAND_ROLL_KP", roll_kp_hind_);
        front_tuck_gain_ = parseEnv("LITE3_HIND_STAND_FRONT_TUCK_GAIN", front_tuck_gain_);
        front_hipy_min_tuck_ = parseEnv("LITE3_HIND_STAND_FRONT_HIPY_MIN", front_hipy_min_tuck_);
        front_knee_min_tuck_ = parseEnv("LITE3_HIND_STAND_FRONT_KNEE_MIN", front_knee_min_tuck_);
        float pitch_sign = parseEnv("LITE3_HIND_STAND_PITCH_SIGN", imu_pitch_sign_);
        imu_pitch_sign_ = (pitch_sign < 0.f) ? -1.f : 1.f;
        if (const char* env = std::getenv("LITE3_HIND_STAND_FEEDBACK")) {
            posture_feedback_enable_ = std::atoi(env) != 0;
        }
        if (const char* env = std::getenv("LITE3_HIND_STAND_ALLOW_JOINT_OVERRIDE")) {
            allow_env_joint_override_ = std::atoi(env) != 0;
        }
        if (const char* env = std::getenv("LITE3_HIND_STAND_AUTO_SIGN")) {
            pitch_auto_sign_flip_ = std::atoi(env) != 0;
        }
    }

    void MaybeLoadDurationFromEnv(){
        const char* env = std::getenv("LITE3_HIND_STAND_DURATION");
        if (!env) return;
        char* endptr = nullptr;
        float parsed = std::strtof(env, &endptr);
        if (endptr == env || !std::isfinite(parsed) || parsed <= 0.f) {
            std::cerr << "[HindStandState] Ignore LITE3_HIND_STAND_DURATION: invalid value\n";
            return;
        }
        transition_duration_ = parsed;
        std::cout << "[HindStandState] transition_duration = " << transition_duration_ << " s\n";
    }

    void BuildJointLimits(){
        joint_pos_lower_ = VecXf::Zero(12);
        joint_pos_upper_ = VecXf::Zero(12);
        Vec3f fl_lower = cp_ptr_->fl_joint_lower_;
        Vec3f fl_upper = cp_ptr_->fl_joint_upper_;
        Vec3f fr_lower(-fl_upper(0), fl_lower(1), fl_lower(2));
        Vec3f fr_upper(-fl_lower(0), fl_upper(1), fl_upper(2));
        joint_pos_lower_ << fl_lower, fr_lower, fl_lower, fr_lower;
        joint_pos_upper_ << fl_upper, fr_upper, fl_upper, fr_upper;
    }

    void ClampGoalToLimits(){
        for(int i=0;i<goal_joint_pos_.rows();++i){
            goal_joint_pos_(i) = LimitNumber(goal_joint_pos_(i), joint_pos_lower_(i), joint_pos_upper_(i));
        }
    }

    void ConfigureGains(){
        kp_ = cp_ptr_->swing_leg_kp_.replicate(4, 1);
        kd_ = cp_ptr_->swing_leg_kd_.replicate(4, 1);

        // Soften front legs so they do not aggressively push when tucked.
        for (int idx : {1, 2, 4, 5}) {
            kp_(idx) = std::min(kp_(idx), 90.f);
            kd_(idx) = std::min(kd_(idx), 2.0f);
        }

        // Hind legs carry body load in this mode.
        for (int idx : {7, 8, 10, 11}) {
            kp_(idx) = std::max(kp_(idx), 200.f);
            kd_(idx) = std::max(kd_(idx), 4.0f);
        }
    }

    void ApplyPostureFeedback(VecXf& target_joint_pos){
        const Vec3f rpy = ri_ptr_->GetImuRpy();
        const float measured_pitch = imu_pitch_sign_ * rpy(1);
        const float pitch_err = NormalizeAngle(torso_pitch_target_ - measured_pitch);
        const float roll_err = NormalizeAngle(-rpy(0));

        if (pitch_auto_sign_flip_ && !one_shot_sign_check_done_ &&
            run_time_ - time_stamp_record_ > transition_duration_ + 0.20f) {
            const float err_abs = std::fabs(pitch_err);
            if (err_abs > enter_pitch_err_abs_ + 0.03f) {
                pitch_gain_sign_ *= -1.f;
                std::cout << "[HindStandState] one-shot pitch sign correction -> " << pitch_gain_sign_ << "\n";
            }
            one_shot_sign_check_done_ = true;
        }

        if (pitch_auto_sign_flip_ && run_time_ - time_stamp_record_ > transition_duration_ + 0.35f) {
            const float err_abs = std::fabs(pitch_err);
            if (err_abs > prev_pitch_err_abs_ + 0.005f) {
                ++pitch_worsen_count_;
            } else {
                pitch_worsen_count_ = std::max(0, pitch_worsen_count_ - 1);
            }
            prev_pitch_err_abs_ = err_abs;
            if (pitch_worsen_count_ > 80) {
                pitch_gain_sign_ *= -1.f;
                pitch_worsen_count_ = 0;
                std::cout << "[HindStandState] auto-flip pitch control sign -> " << pitch_gain_sign_ << "\n";
            }
        }

        const float hind_hy_delta = LimitNumber(pitch_gain_sign_ * pitch_kp_hipy_ * pitch_err, -0.25f, 0.25f);
        const float hind_knee_delta = LimitNumber(pitch_gain_sign_ * pitch_kp_knee_ * pitch_err, -0.45f, 0.45f);
        const float hind_roll_delta = LimitNumber(roll_kp_hind_ * roll_err, -0.22f, 0.22f);

        // Hind legs: drive torso pitch and keep lateral balance.
        target_joint_pos(7)  += hind_hy_delta + hind_roll_delta;
        target_joint_pos(8)  += hind_knee_delta + 0.5f * hind_roll_delta;
        target_joint_pos(10) += hind_hy_delta - hind_roll_delta;
        target_joint_pos(11) += hind_knee_delta - 0.5f * hind_roll_delta;

        // Front legs: maintain tucked, disconnected posture while torso rises.
        const float front_hy_delta = LimitNumber(-front_tuck_gain_ * hind_hy_delta, -0.30f, 0.30f);
        const float front_knee_delta = LimitNumber(-front_tuck_gain_ * hind_knee_delta, -0.45f, 0.45f);
        target_joint_pos(1) += front_hy_delta;
        target_joint_pos(2) += front_knee_delta;
        target_joint_pos(4) += front_hy_delta;
        target_joint_pos(5) += front_knee_delta;
        target_joint_pos(1) = std::max(target_joint_pos(1), front_hipy_min_tuck_);
        target_joint_pos(4) = std::max(target_joint_pos(4), front_hipy_min_tuck_);
        target_joint_pos(2) = std::max(target_joint_pos(2), front_knee_min_tuck_);
        target_joint_pos(5) = std::max(target_joint_pos(5), front_knee_min_tuck_);

        // Respect hardware/sim joint limits after feedback shaping.
        for (int i = 0; i < target_joint_pos.rows(); ++i) {
            target_joint_pos(i) = LimitNumber(target_joint_pos(i), joint_pos_lower_(i), joint_pos_upper_(i));
        }

        if (last_debug_print_time_ < 0.f || run_time_ - last_debug_print_time_ > 0.5f) {
            std::cout << "[HindStandState] rpy(deg)= " << (180.f/M_PI)*rpy.transpose()
                      << " pitch_err(deg)= " << (180.f/M_PI)*pitch_err
                      << " sign=" << pitch_gain_sign_
                      << " imu_pitch_sign=" << imu_pitch_sign_
                      << " q_hind(hy,knee)= " << target_joint_pos(7) << "," << target_joint_pos(8)
                      << " | " << target_joint_pos(10) << "," << target_joint_pos(11) << "\n";
            last_debug_print_time_ = run_time_;
        }
    }

public:
    HindStandState(const RobotType& robot_type, const std::string& state_name,
        std::shared_ptr<ControllerData> data_ptr):StateBase(robot_type, state_name, data_ptr){
            goal_joint_pos_ = VecXf::Zero(12);
            // Explicit "sit-up / hind-legs-only" target:
            // front legs strongly tucked, hind legs folded under torso.
            goal_joint_pos_ << -0.03f,  0.30f, 2.70f,
                                0.03f,  0.30f, 2.70f,
                               -0.08f, -0.18f, 2.20f,
                                0.08f, -0.18f, 2.20f;
            BuildJointLimits();
            MaybeLoadPostureFeedbackFromEnv();
            TryLoadJointTargetFromEnv();
            ClampGoalToLimits();
            ConfigureGains();
            joint_cmd_ = MatXf::Zero(12, 5);
            joint_cmd_.col(0) = kp_;
            joint_cmd_.col(2) = kd_;
            transition_duration_ = std::max(2.8f, cp_ptr_->stand_duration_);
            MaybeLoadDurationFromEnv();
            std::cout << "[HindStandState] target joints: " << goal_joint_pos_.transpose() << "\n";
            std::cout << "[HindStandState] pitch_target=" << torso_pitch_target_
                      << " pitch_kp(hipy/knee)=" << pitch_kp_hipy_ << "/" << pitch_kp_knee_
                      << " roll_kp=" << roll_kp_hind_
                      << " pitch_sign=" << imu_pitch_sign_
                      << " front_tuck_min(hipy/knee)=" << front_hipy_min_tuck_ << "/" << front_knee_min_tuck_
                      << " front_tuck_gain=" << front_tuck_gain_
                      << " allow_joint_override=" << (allow_env_joint_override_ ? 1 : 0)
                      << " feedback=" << (posture_feedback_enable_ ? 1 : 0)
                      << " auto_sign=" << (pitch_auto_sign_flip_ ? 1 : 0) << "\n";
        }

    ~HindStandState(){}

    virtual void OnEnter() {
        GetRobotJointValue();
        RecordJointData();
        pitch_gain_sign_ = 1.f;
        enter_pitch_err_abs_ = std::fabs(NormalizeAngle(torso_pitch_target_ - imu_pitch_sign_ * ri_ptr_->GetImuRpy()(1)));
        prev_pitch_err_abs_ = 1e6f;
        pitch_worsen_count_ = 0;
        one_shot_sign_check_done_ = false;
        last_debug_print_time_ = -1.f;
        StateBase::msfb_.UpdateCurrentState(RobotMotionState::HindStand);
        uc_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };

    virtual void OnExit() {}

    virtual void Run() {
        GetRobotJointValue();
        VecXf planning_joint_pos(current_joint_pos_.rows());
        VecXf planning_joint_vel(current_joint_pos_.rows());
        const float elapsed = run_time_ - time_stamp_record_;
        if (elapsed < transition_duration_) {
            for(int i=0;i<current_joint_pos_.rows();++i){
                planning_joint_pos(i) = GetCubicSplinePos(init_joint_pos_(i), init_joint_vel_(i), goal_joint_pos_(i), 0,
                                                        elapsed, transition_duration_);
                planning_joint_vel(i) = GetCubicSplineVel(init_joint_pos_(i), init_joint_vel_(i), goal_joint_pos_(i), 0,
                                                        elapsed, transition_duration_);
            }
        } else {
            planning_joint_pos = goal_joint_pos_;
            planning_joint_vel.setZero();
            if (posture_feedback_enable_) {
                ApplyPostureFeedback(planning_joint_pos);
            }
        }
        joint_cmd_.col(1) = planning_joint_pos;
        joint_cmd_.col(3) = planning_joint_vel;
        ri_ptr_->SetJointCommand(joint_cmd_);
    }

    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::JointDamping)) return true;
        return false;
    }

    virtual StateName GetNextStateName() {
        if(run_time_ - time_stamp_record_ < transition_duration_){
            return StateName::kHindStand;
        }
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::StandingUp)){
            return StateName::kStandUp;
        }
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::RLControlMode)){
            return StateName::kRLControl;
        }
        return StateName::kHindStand;
    }
};

#endif
