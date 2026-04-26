"""
Gymnasium-compatible environment for the T1 humanoid robot in PyBullet.
Observation space: torso pose, velocities, joint angles, joint velocities.
Action space: continuous torques for actuated joints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

from robot import T1
from config import ENV_CONFIG, REWARD_WEIGHTS, ACTUATED_JOINT_INDICES, ARM_JOINT_INDICES


class T1WalkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, reward_weights=None, mask_arms=False):
        super().__init__()

        self.render_mode = render_mode
        self.cfg = ENV_CONFIG
        self.reward_weights = reward_weights if reward_weights is not None else REWARD_WEIGHTS
        self.actuated_indices = ACTUATED_JOINT_INDICES
        self.num_actuated = len(self.actuated_indices)

        # Curriculum: when True, arm action components are overridden to hold
        # arms at their neutral (zero) position. Used for phase-1 training.
        self.mask_arms = mask_arms
        # Indices *within the action vector* that correspond to arm joints.
        self._arm_action_indices = [
            i for i, idx in enumerate(self.actuated_indices)
            if idx in ARM_JOINT_INDICES
        ]

        # Physics sub-steps per policy step
        self.sim_steps_per_action = self.cfg["control_freq"] // self.cfg["policy_freq"]

        # ── Observation space ────────────────────────────────────────────────
        # torso z (1) + torso orientation as euler (3) + torso linear vel (3)
        # + torso angular vel (3) + joint positions (N actuated) + joint velocities (N actuated)
        obs_dim = 1 + 3 + 3 + 3 + self.num_actuated + self.num_actuated
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        # ── Action space: normalized torques in [-1, 1] ─────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.num_actuated,), dtype=np.float32)

        # Will be set in reset()
        self.client = None
        self.robot = None
        self.plane = None
        self.step_count = 0
        self.prev_x = 0.0

    def set_mask_arms(self, mask):
        """Enable/disable curriculum arm masking (callable from VecEnv.env_method)."""
        self.mask_arms = bool(mask)

    def _is_connected(self):
        """Check if the physics client is still alive."""
        if self.client is None:
            return False
        try:
            p.getConnectionInfo(self.client)
            return True
        except Exception:
            return False

    def _connect(self):
        if self._is_connected():
            return
        # Reset state if we lost connection
        self.client = None
        self.robot = None
        self.plane = None

        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=40.0,
                                         cameraPitch=-20.0,
                                         cameraTargetPosition=[0, 0, 1],
                                         physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.cfg["control_freq"], physicsClientId=self.client)

        urdf_root = pybullet_data.getDataPath()
        self.plane = p.loadURDF(os.path.join(urdf_root, "plane.urdf"),
                                basePosition=[0, 0, 0],
                                physicsClientId=self.client)

    def _do_reset(self):
        """Internal reset logic, may raise if server dies."""
        self._connect()

        if self.robot is None:
            # First reset (or after reconnect): load the URDF
            self.robot = T1(
                basePosition=[0, 0, self.cfg["initial_height"]],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                physicsClient=self.client,
            )
            # Cache joint limits for action scaling
            self._joint_lower = self.robot.joint_lower[self.actuated_indices]
            self._joint_upper = self.robot.joint_upper[self.actuated_indices]
            self._joint_mid = (self._joint_lower + self._joint_upper) / 2.0
            self._joint_range = (self._joint_upper - self._joint_lower) / 2.0
        else:
            # Subsequent resets: move robot back to origin
            self.robot.reset_base(
                [0, 0, self.cfg["initial_height"]],
                p.getQuaternionFromEuler([0, 0, 0]),
            )

        self.robot.reset([0.0] * T1.NUM_JOINTS)

        # Settle with motors holding the standing pose (not ragdolling)
        max_forces = self.robot.joint_max_torque[self.actuated_indices].tolist()
        zero_targets = [0.0] * self.num_actuated
        p.setJointMotorControlArray(
            self.robot.robot, self.actuated_indices, p.POSITION_CONTROL,
            targetPositions=zero_targets,
            positionGains=[0.5] * self.num_actuated,
            velocityGains=[1.0] * self.num_actuated,
            forces=max_forces,
            physicsClientId=self.client)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)

        self.step_count = 0
        state = self.robot.get_state()
        self.prev_x = state["base-position"][0]
        return self._build_obs(state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for attempt in range(3):
            try:
                obs = self._do_reset()
                return obs, {}
            except Exception:
                # Server died — force full reconnect
                self.client = None
                self.robot = None
                self.plane = None
        # If all retries fail, return zeros
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # If physics server died (user closed GUI), reconnect and reset
        if not self._is_connected():
            self.client = None
            self.robot = None
            self.plane = None
            obs, info = self.reset()
            return obs, -10.0, True, False, info

        action = np.clip(action, -1.0, 1.0)

        # Curriculum phase 1: mask arm actions → hold arms at neutral (zero) position.
        # The network still outputs 21 actions; we just override the arm components.
        target_positions = self._joint_mid + action * self._joint_range
        if self.mask_arms and self._arm_action_indices:
            for i in self._arm_action_indices:
                target_positions[i] = 0.0  # override to physical zero, not joint midpoint

        # Apply position control with limited forces (covers all 21 actuated joints)
        max_forces = self.robot.joint_max_torque[self.actuated_indices].tolist()
        try:
            p.setJointMotorControlArray(
                self.robot.robot, self.actuated_indices, p.POSITION_CONTROL,
                targetPositions=target_positions.tolist(),
                positionGains=[0.2] * self.num_actuated,
                velocityGains=[0.5] * self.num_actuated,
                forces=max_forces,
                physicsClientId=self.client)

            for _ in range(self.sim_steps_per_action):
                p.stepSimulation(physicsClientId=self.client)
        except Exception:
            # Server died mid-step, terminate episode
            self.client = None
            self.robot = None
            self.plane = None
            obs, info = self.reset()
            return obs, -10.0, True, False, info

        self.step_count += 1
        state = self.robot.get_state()

        # Get applied torques for energy computation
        applied_torques = state["joint-torque"][self.actuated_indices]

        # ── Compute reward ───────────────────────────────────────────────────
        reward, reward_info = self._compute_reward(state, applied_torques)

        # ── Check termination ────────────────────────────────────────────────
        torso_z = state["base-position"][2]
        euler = p.getEulerFromQuaternion(state["base-orientation"])
        roll, pitch = abs(euler[0]), abs(euler[1])
        fallen = (torso_z < self.cfg["fall_threshold"] or
                  torso_z > 2.0 or  # robot launched into air
                  roll > 1.2 or pitch > 1.2)  # torso tilted > ~70 deg
        truncated = self.step_count >= self.cfg["max_episode_steps"]
        terminated = fallen

        if fallen:
            reward += self.reward_weights["fall_penalty"]

        self.prev_x = state["base-position"][0]

        obs = self._build_obs(state)
        info = {"reward_info": reward_info,
                "torso_z": torso_z,
                "x_distance": state["base-position"][0]}
        return obs, reward, terminated, truncated, info

    def _build_obs(self, state):
        """Build flat observation vector from robot state."""
        torso_z = np.array([state["base-position"][2]])
        euler = np.array(p.getEulerFromQuaternion(state["base-orientation"]))
        lin_vel = state["base-linear-velocity"]
        ang_vel = state["base-angular-velocity"]
        joint_pos = state["joint-position"][self.actuated_indices]
        joint_vel = state["joint-velocity"][self.actuated_indices]

        obs = np.concatenate([torso_z, euler, lin_vel, ang_vel,
                              joint_pos, joint_vel]).astype(np.float32)
        return obs

    def _compute_reward(self, state, torques):
        w = self.reward_weights

        # 1. Forward velocity reward — linear in v_x, grows without bound.
        #    Goal: push the robot to walk as fast as possible.
        current_x = state["base-position"][0]
        forward_vel = (current_x - self.prev_x) * self.cfg["policy_freq"]
        vel_reward = w["forward_velocity"] * forward_vel

        # 2. Survival bonus
        survival = w["survival"]

        # 3. Energy penalty (average power per joint, not total)
        joint_vel = state["joint-velocity"][self.actuated_indices]
        energy = np.mean(np.abs(torques * joint_vel))
        energy_pen = w["energy_penalty"] * energy

        # 4. Orientation penalty (keep torso upright)
        euler = p.getEulerFromQuaternion(state["base-orientation"])
        roll, pitch = euler[0], euler[1]
        orientation_pen = w["orientation_penalty"] * (roll ** 2 + pitch ** 2)

        # 5. Joint limit penalty (gradual quadratic ramp starting at 50% of range)
        joint_pos = state["joint-position"][self.actuated_indices]
        lower = self.robot.joint_lower[self.actuated_indices]
        upper = self.robot.joint_upper[self.actuated_indices]
        ranges = upper - lower + 1e-8
        # Normalized distance from center (0 = center, 1 = at limit)
        normalized = 2.0 * (joint_pos - lower) / ranges - 1.0
        # Quadratic penalty past 50%: gentle at 0.5, strong near 1.0
        excess = np.maximum(np.abs(normalized) - 0.5, 0.0)
        limit_pen = w["joint_limit_penalty"] * np.sum(excess ** 2)

        # 6. Height reward: reward torso z being close to initial height
        torso_z = state["base-position"][2]
        target_z = self.cfg["initial_height"]
        # Gaussian reward centered on initial height, drops off as it deviates
        height_rew = w.get("height_reward", 0) * np.exp(-5.0 * (torso_z - target_z) ** 2)

        # 7. Z-velocity penalty: penalize upward z velocity — max(z_dot, 0).
        #    Discourages hopping/launching; coefficient is negative so reward is negative.
        z_vel = state["base-linear-velocity"][2]
        z_vel_pen = w.get("z_velocity_penalty", 0) * max(z_vel, 0.0)

        total_reward = (vel_reward + survival + energy_pen + orientation_pen
                        + limit_pen + height_rew + z_vel_pen)

        reward_info = {
            "velocity_reward": vel_reward,
            "survival_reward": survival,
            "energy_penalty": energy_pen,
            "orientation_penalty": orientation_pen,
            "joint_limit_penalty": limit_pen,
            "height_reward": height_rew,
            "z_velocity_penalty": z_vel_pen,
            "forward_vel": forward_vel,
        }
        return total_reward, reward_info

    def render(self):
        if self.render_mode == "rgb_array":
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 1], distance=3.0,
                yaw=40, pitch=-20, roll=0, upAxisIndex=2,
                physicsClientId=self.client)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=100,
                physicsClientId=self.client)
            _, _, img, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                physicsClientId=self.client)
            return np.array(img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return None

    def close(self):
        if self.client is not None:
            try:
                p.disconnect(self.client)
            except Exception:
                pass
            self.client = None
            self.robot = None


# Register the environment with Gymnasium
gym.register(
    id="T1Walking-v0",
    entry_point="env:T1WalkingEnv",
    max_episode_steps=ENV_CONFIG["max_episode_steps"],
)
