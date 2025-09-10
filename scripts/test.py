import sys
print(sys.executable)

import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from utils import ZEDCamera
import rtde_control
import rtde_receive

side = "right"
if side == "right":
    rtde_c = rtde_control.RTDEControlInterface("10.0.0.219")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.219")
elif side == "left":
    rtde_c = rtde_control.RTDEControlInterface("10.0.0.78")
    rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.78")

# Initialize camera 
camera = ZEDCamera()
print("Camera initialized")

# Discarding first two frames from realsense 
for _ in range(50):
    rgb = camera.get_rgb_image()

plt.imshow(rgb)
plt.show()