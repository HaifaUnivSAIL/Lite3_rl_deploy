
#include "state_machine.hpp"
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>
#ifdef BUILD_SIMULATION
    #define BACKWARD_HAS_DW 1
    #include "backward.hpp"
    namespace backward{
        backward::SignalHandling sh;
    }
#endif
using namespace types;

MotionStateFeedback StateBase::msfb_ = MotionStateFeedback();

static void PrintActiveBehaviorOverrides() {
    const std::vector<std::pair<const char*, const char*>> overrides = {
        {"LITE3_FORCE_RL_START", "idle -> rl_control auto transition"},
        {"LITE3_FIXED_CMD", "fixed command override in RL state"},
        {"LITE3_DEFAULT_CMD", "default RL command preload"},
        {"LITE3_DEPLOY_STATE", "deterministic deploy reset snapshot"},
        {"LITE3_HISTORY_SEED_FILE", "history replay seed override"},
        {"LITE3_HISTORY_SEED_MODE", "history seed mode override"},
        {"LITE3_DISABLE_POSTURE_CHECK", "posture safety guard disable"},
        {"LITE3_POLICY_ASYNC", "async policy execution mode override"},
        {"LITE3_POLICY_DECIMATION", "policy decimation override"},
        {"LITE3_POLICY_CONTROL_DT", "policy control dt override"},
        {"LITE3_MUJOCO_DT", "sim dt override"},
        {"LITE3_RANDOM_RESET", "random reset override"},
        {"LITE3_IMU_GYRO_NOISE_STD", "IMU gyro noise override"},
        {"LITE3_IMU_RPY_NOISE_STD", "IMU rpy noise override"},
        {"LITE3_IMU_ACC_NOISE_STD", "IMU acc noise override"},
    };

    bool any_active = false;
    for (const auto& item : overrides) {
        const char* value = std::getenv(item.first);
        if (value && *value != '\0') {
            if (!any_active) {
                std::cout << "[CONFIG] Active behavior overrides detected:\n";
                any_active = true;
            }
            std::cout << "  - " << item.first << "=" << value
                      << " (" << item.second << ")\n";
        }
    }
    if (!any_active) {
        std::cout << "[CONFIG] No behavior overrides active. Running default control flow.\n";
    }
}

int main(){
    PrintActiveBehaviorOverrides();
    StateMachine state_machine(RobotType::Lite3);
    state_machine.Run();
    return 0;
}
