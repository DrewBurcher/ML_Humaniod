"""
Hyperparameter and reward configuration for the T1 bipedal locomotion project.
"""

# ── Environment ──────────────────────────────────────────────────────────────
ENV_CONFIG = {
    "control_freq": 240,           # PyBullet simulation frequency (Hz)
    "policy_freq": 60,             # How often the policy acts (Hz)
    "max_episode_steps": 1000,     # Steps per episode at policy_freq (~33 s)
    "initial_height": 0.85,        # Base spawn height (m)
    "fall_threshold": 0.35,        # Torso z below this → fallen (m)
    "target_speed": 0.5,           # Desired forward speed (m/s)
}

# ── Reward weights (tunable) ─────────────────────────────────────────────────
REWARD_WEIGHTS = {
    "forward_velocity": 1.5,       # Reward for moving toward target speed (1.5x)
    "survival": 2.0,               # Bonus each timestep for staying alive
    "energy_penalty": -0.0033,     # Penalty per avg power (mean |torque*vel|)
    "fall_penalty": -100.0,        # Large penalty for falling
    "orientation_penalty": -1.0,   # Penalty for torso tilt (quadratic on roll+pitch)
    "joint_limit_penalty": -2.0,   # Gradual quadratic penalty past 50% of range
    "height_reward": 1.0,          # Reward for maintaining torso height near initial
    "z_fall_velocity_penalty": -0.5,  # Penalty for downward z velocity (don't reward upward)
}

# ── PPO hyperparameters ──────────────────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "device": "cpu",               # CPU is faster than GPU for small MLPs + PyBullet
    "total_timesteps": 2_000_000,
}

# ── SAC hyperparameters ──────────────────────────────────────────────────────
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "target_entropy": -5,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "device": "cpu",
    "total_timesteps": 2_000_000,
}

# ── Actuated joints (indices into the 23-joint list) ─────────────────────────
# We skip head joints (0, 1) as they don't contribute to locomotion.
# Indices: Waist(10), Left leg(11-16), Right leg(17-22), Arms(2-9)
LEG_JOINT_INDICES = [
    10,                   # Waist
    11, 12, 13, 14, 15, 16,  # Left leg
    17, 18, 19, 20, 21, 22,  # Right leg
]

ARM_JOINT_INDICES = [
    2, 3, 4, 5,    # Left arm
    6, 7, 8, 9,    # Right arm
]

# Active actuated joints — legs only for now (arms added later)
ACTUATED_JOINT_INDICES = LEG_JOINT_INDICES
