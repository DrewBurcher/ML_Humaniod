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

    def __init__(self, basePosition, baseOrientation, jointStartPositions=None):
        urdf_path = os.path.join(os.path.dirname(__file__), "T1", "T1_serial.urdf")
        self.robot = p.loadURDF(urdf_path,
                                basePosition=basePosition,
                                baseOrientation=baseOrientation,
                                useFixedBase=False)
        if jointStartPositions is not None:
            self.reset(jointStartPositions)

    def reset(self, jointStartPositions):
        for idx in range(len(jointStartPositions)):
            p.resetJointState(self.robot, idx, jointStartPositions[idx])

    def get_state(self):
        joint_values = p.getJointStates(self.robot, range(self.NUM_JOINTS))
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        state = {}
        state["base-position"] = base_pos
        state["base-orientation"] = base_orn
        state["joint-position"] = [item[0] for item in joint_values]
        state["joint-velocity"] = [item[1] for item in joint_values]
        state["joint-torque"] = [item[3] for item in joint_values]
        return state

    def set_joint_positions(self, joint_indices, target_positions, positionGains=None):
        if positionGains is None:
            positionGains = [1.0] * len(joint_indices)
        p.setJointMotorControlArray(self.robot, joint_indices, p.POSITION_CONTROL,
                                    targetPositions=target_positions,
                                    positionGains=positionGains)
