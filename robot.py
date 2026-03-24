import pybullet as p
import numpy as np
import os


class T1():

    JOINT_NAMES = [
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ]
    NUM_JOINTS = 23

    def __init__(self, basePosition, baseOrientation, physicsClient=None, jointStartPositions=None):
        self.client = physicsClient
        urdf_path = os.path.join(os.path.dirname(__file__), "T1", "T1_serial.urdf")
        self.robot = p.loadURDF(urdf_path,
                                basePosition=basePosition,
                                baseOrientation=baseOrientation,
                                useFixedBase=False,
                                physicsClientId=self.client) if self.client is not None else \
                     p.loadURDF(urdf_path,
                                basePosition=basePosition,
                                baseOrientation=baseOrientation,
                                useFixedBase=False)

        # Read joint limits and max torques from URDF
        self.joint_lower = np.zeros(self.NUM_JOINTS)
        self.joint_upper = np.zeros(self.NUM_JOINTS)
        self.joint_max_torque = np.zeros(self.NUM_JOINTS)
        self.joint_max_velocity = np.zeros(self.NUM_JOINTS)
        for i in range(self.NUM_JOINTS):
            info = p.getJointInfo(self.robot, i,
                                  physicsClientId=self.client) if self.client is not None else \
                   p.getJointInfo(self.robot, i)
            self.joint_lower[i] = info[8]
            self.joint_upper[i] = info[9]
            self.joint_max_torque[i] = info[10]
            self.joint_max_velocity[i] = info[11]

        if jointStartPositions is not None:
            self.reset(jointStartPositions)

    def _pb(self):
        """Return kwargs for physicsClientId if we have one."""
        if self.client is not None:
            return {"physicsClientId": self.client}
        return {}

    def reset(self, jointStartPositions):
        for idx in range(len(jointStartPositions)):
            p.resetJointState(self.robot, idx, jointStartPositions[idx], **self._pb())

    def reset_base(self, position, orientation):
        p.resetBasePositionAndOrientation(self.robot, position, orientation, **self._pb())
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0], **self._pb())

    def get_state(self):
        joint_values = p.getJointStates(self.robot, range(self.NUM_JOINTS), **self._pb())
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot, **self._pb())
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot, **self._pb())
        state = {}
        state["base-position"] = np.array(base_pos)
        state["base-orientation"] = np.array(base_orn)
        state["base-linear-velocity"] = np.array(base_lin_vel)
        state["base-angular-velocity"] = np.array(base_ang_vel)
        state["joint-position"] = np.array([item[0] for item in joint_values])
        state["joint-velocity"] = np.array([item[1] for item in joint_values])
        state["joint-torque"] = np.array([item[3] for item in joint_values])
        return state

    def set_joint_positions(self, joint_indices, target_positions, positionGains=None):
        if positionGains is None:
            positionGains = [1.0] * len(joint_indices)
        p.setJointMotorControlArray(self.robot, joint_indices, p.POSITION_CONTROL,
                                    targetPositions=target_positions,
                                    positionGains=positionGains,
                                    **self._pb())

    def apply_torques(self, joint_indices, torques):
        """Apply torque control to specified joints, clamped to max torque."""
        clamped = []
        for i, idx in enumerate(joint_indices):
            max_t = self.joint_max_torque[idx]
            clamped.append(np.clip(torques[i], -max_t, max_t))
        p.setJointMotorControlArray(self.robot, joint_indices, p.TORQUE_CONTROL,
                                    forces=clamped,
                                    **self._pb())

    def disable_default_motors(self, joint_indices):
        """Disable default velocity motors so torque control works properly."""
        zero_forces = [0.0] * len(joint_indices)
        zero_velocities = [0.0] * len(joint_indices)
        p.setJointMotorControlArray(self.robot, joint_indices, p.VELOCITY_CONTROL,
                                    targetVelocities=zero_velocities,
                                    forces=zero_forces,
                                    **self._pb())
