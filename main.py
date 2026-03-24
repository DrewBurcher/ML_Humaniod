import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from robot import T1

# parameters
control_dt = 1. / 240.

# create simulation
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                cameraYaw=40.0,
                                cameraPitch=-20.0,
                                cameraTargetPosition=[0.0, 0.0, 1.0])

# load ground plane
urdfRootPath = pybullet_data.getDataPath()
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, 0])

# load the T1 robot
robot = T1(basePosition=[0, 0, 0.85],
           baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

for idx in range(10000):

    robot_state = robot.get_state()

    p.stepSimulation()
    time.sleep(control_dt)
