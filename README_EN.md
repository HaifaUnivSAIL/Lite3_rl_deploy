[简体中文](./README.md)

## Sim-to-Sim

```bash
# segmentation debug tools install
sudo apt-get install libdw-dev
wget https://raw.githubusercontent.com/bombela/backward-cpp/master/backward.hpp
sudo mv backward.hpp /usr/include

# Dependency install (python3.10)
pip install pybullet "numpy < 2.0" mujoco
git clone --recurse-submodule https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git

# compile
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON -DSEND_REMOTE=OFF

# Explanation
# -DBUILD_PLATFORM：device platform，Ubuntu is x86，quadruped is arm
# -DBUILD_SIM：whether or not to use simulatior, if deployed on real robots, set to OFF 
make -j
```

```bash
# run (open 2 terminals)
# Terminal 1 (pybullet)
cd interface/robot/simulation
python pybullet_simulation.py

# Terminal 1 (mujoco)
cd interface/robot/simulation
python mujoco_simulation.py

# Terminal 2 
cd build
./rl_deploy
```

Default timing note:

- In legacy UDP simulation mode (`BUILD_SIM=ON`, `USE_PYBULLET=ON`), deploy now defaults to a `0.001s` interface period assumption and `decimation=20`, so policy control period is `0.02s` (aligned with training: decimation=4, sim.dt=0.005).
- For debug overrides you can still use `LITE3_MUJOCO_DT` and `LITE3_POLICY_DECIMATION`.
- A startup timing-parity assert is enforced (training vs deploy control period). If training timing was intentionally changed, override the assert baseline with `LITE3_TRAINING_SIM_DT` and `LITE3_TRAINING_DECIMATION`.

## Usage(Terminal 2)

tips：right click simulator window and select "always on top"

- z： default position
- x： switch to hind-leg stand from stand-up state
- c： rl control default position
- wasd：forward/leftward/backward/rightward
- qe：clockwise/counter clockwise

change quadruped ip: go to jy_exe/conf/network.toml, change ip to 192.168.2.1
# Sim-to-Real
This process is almost identical to simulation-simulation. You only need to add the step of connecting to Wi-Fi to transfer data, and then modify the compilation instructions. Currently, the default real-machine control mode is Retroid controller mode. If you need to use keyboard mode, you can change state_machine/state_machine.hpp line121 to
```bash
uc_ptr_ = std::make_shared<KeyboardInterface>();
```
```bash
# apply code_modification

# computer and gamepad should both connect to WiFi
# WiFi: Lite*******
# Passward: 12345678 (If wrong, contact technical support)

# scp to transfer files to quadruped (open a terminal on your local computer)
scp -r ~/Lite3_rl_deploy ysc@192.168.2.1:~/

# ssh connect for remote development, the passward has the following 3 combinations
#Username	Password
#ysc		' (a single quote)
#user		123456 (Recommended)
#firefly	firefly
ssh ysc@192.168.2.1
# enter your passward, the terminal will be active on the qurdruped computer

# compile
cd Lite3_rl_deploy
mkdir build && cd build
cmake .. -DBUILD_PLATFORM=arm -DBUILD_SIM=OFF -DSEND_REMOTE=OFF 
# Explanation
# -DBUILD_PLATFORM：device platform，Ubuntu is x86，quadruped is arm
# -DBUILD_SIM：whether or not to use simulatior, if deployed on real robots, set to OFF 
make -j
./rl_deploy
```

## Usage(Retroid gamepad)

Please refer to https://github.com/DeepRoboticsLab/gamepad

## Model Conversion

To run the policy file trained with RL, you need to link the onnxruntime library, which supports models in the .onnx format. Therefore, you must manually convert the .pt model to the .onnx format.

You can convert the .pt model to the .onnx model by running the pt2onnx.py file in the policy folder. Pay attention to the program output to compare the consistency between the two models.

First, configure and verify the program runtime environment:

```bash
pip install torch numpy onnx onnxruntime

python3 -c 'import torch, numpy, onnx, onnxruntime; print(" All modules OK")'
```

Then, run the program:

```bash
cd Lite3_rl_deploy/policy/

# Export checkpoint -> ONNX (deploy expects 117 obs and 40 history frames for two-leg stand)
python pt2onnx.py \
  --ckpt /path/to/model_7000.pt \
  --out ppo/policy.onnx \
  --num-obs 117 \
  --history-len 40
```

Afterward, the deploy binary will load `Lite3_rl_deploy/policy/ppo/policy.onnx` by default.

Optional override (no rebuild): set `LITE3_POLICY_ONNX` to an absolute path, or a path relative to `Lite3_rl_deploy/build`.

### Debug parity (training vs deploy)

To compare observations/actions against the training stack, use the debug dumps under `Lite3_rl_training/debug_training_obs`:

```bash
# Training-side (writes step_00.npz, ... into Lite3_rl_training/debug_training_obs)
python Lite3_rl_training/legged_gym/legged_gym/scripts/play.py --task lite3_two_leg_stand --load_run <run_name>

# Deploy-side (writes debug_cpp_step0.txt, ... into Lite3_rl_training/debug_training_obs)
export LITE3_DEBUG_DUMPS=10
./Lite3_rl_deploy/build/rl_deploy

# Compare
python Lite3_rl_training/debug_training_obs/compare_blocks.py --root Lite3_rl_training/debug_training_obs --steps 10
```

### state_machine


```mermaid
graph LR
A(Idle) -->B(StandUp) --> C(RL) 
C-->D(JointDamping)
B-->D
D-->A

```

The state_machine module is where Lite3 switches between different states, the different states represent the following functions:

1.Idle : Idle state, indicating that the robot is in a situation where the joints do not enabled.

2.StandUp : Stand up state, indicating the action of the robot dog from sit to stand.

3.RL : RL control state，indicating the action output by the robot execution strategy.

4.JointDamping : Joint damping state, indicating that the joints of the robot are in the damping control state

Note: the transition `RL -> JointDamping` can be triggered automatically by the posture safety guard (roll/pitch limits).
You can disable it with `LITE3_DISABLE_POSTURE_CHECK=1`, or tune thresholds via `LITE3_POSTURE_LIMIT_ROLL_DEG` / `LITE3_POSTURE_LIMIT_PITCH_DEG`.

### interface

```mermaid
graph LR
A(Interface) -->B(Robot)
A --> C(User Command)
B-->D(simulation)
B-->E(hardware)
C-->F(gamepad)
C-->G(keyboard)

```

The interface module represents the inputs for the dog's data receiving and sending interface and joystick control. Among them, the inputs of the robot platform are divided into simulation and physical, and the inputs of the controller are divided into keyboard and joystick control.

### run_policy

```mermaid
graph LR
A(policy_runner_base) -->B(policy_runner)


```

This section is used to execute the output of the RL policy, new policies can be implemented by inheriting policy_runner_base.
