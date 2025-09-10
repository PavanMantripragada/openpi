import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import copy
from utils import ZEDCamera

def project_object_poses(gt_R_cl, K, img):
    axes_scale = 0.1
    o = K @ gt_R_cl[:3,3].reshape(3,1)
    o /= o[2]
    px = K @ (gt_R_cl[:3,3].reshape(3,1) + axes_scale*gt_R_cl[:3,0].reshape(3,1))
    px /= px[2]
    py = K @ (gt_R_cl[:3,3].reshape(3,1) + axes_scale*gt_R_cl[:3,1].reshape(3,1))
    py /= py[2]
    pz = K @ (gt_R_cl[:3,3].reshape(3,1) + axes_scale*gt_R_cl[:3,2].reshape(3,1))
    pz /= pz[2]
    # img = rgb[:,:848]
    cv2.line(img, (int(o[0]), int(o[1])), (int(px[0]), int(px[1])), (0, 0, 255), 2)
    cv2.line(img, (int(o[0]), int(o[1])), (int(py[0]), int(py[1])), (0, 255, 0), 2)
    cv2.line(img, (int(o[0]), int(o[1])), (int(pz[0]), int(pz[1])), (255, 0, 0), 2)
    return img

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


# Config and checkpoint of trained model
config = config.get_config("pi0_fast_ego_mimic_low_mem_finetune")
checkpoint_dir = "checkpoints/pi0_fast_ego_mimic_low_mem_finetune/delta_ee_poses_sparse/22000"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

K = np.load("/home/pavan/Documents/Projects/Imitation/scripts/data/right_hand_eye_calib/rgb_intrinsics.npy")
T_c2b = np.load("/home/pavan/Documents/Projects/Imitation/scripts/data/right_hand_eye_calib/camera_to_base.npy")

for i in range(1000):
    image = cv2.imread("/home/pavan/Documents/Projects/openpi/data/image.png")
    # state = np.array(rtde_r.getActualQ())
    ee_pose = rtde_r.getActualTCPPose()
    state = np.zeros((7,))
    state[:6] = np.array(ee_pose[:6])

    state_pose = np.eye(4)
    state_pose[:3, 3] = ee_pose[:3]
    state_pose[:3, :3] = R.from_rotvec(ee_pose[3:6]).as_matrix()
    ee_pose = state_pose

    instruction = "open"
    # print(f"Current state: {type(state)}")
    # print(type(image))
    # Run inference on a dummy example.
    example = {
        "observation/state": state,
        "observation/image": image,
        "observation/wrist_image": image,
        "prompt": instruction,
    }

    action_chunk = policy.infer(example)["actions"]
    # print(f"delta chnage by this action in deg: {180*np.abs(action_chunk[0][:6]-state)/np.pi}")

    cv2.imshow("Predicted Future Poses", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for action in action_chunk:
        delta_pose = np.eye(4)
        delta_pose[:3, 3] = action[:3]
        delta_pose[:3, :3] = R.from_rotvec(action[3:6]).as_matrix()
        print(f"delta pose: {delta_pose}")

        pose = ee_pose @ delta_pose
        image = project_object_poses(pose, K, image)
        ee_pose = copy.deepcopy(pose)
    
    cv2.imshow("Predicted Future Poses", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ans = input("Do you want to execute this action? (y/n): ")
    if ans == 'y':
        # rtde_c.moveJ(action_chunk[0][:6], 0.05)
        delta_pose = np.eye(4)
        delta_pose[:3, 3] = action_chunk[0][:3]
        delta_pose[:3, :3] = R.from_rotvec(action_chunk[0][3:6]).as_matrix()
        ee_pose = rtde_r.getActualTCPPose()
        pose = ee_pose @ delta_pose
        target_pose = np.zeros((6,))
        target_pose[:3] = pose[:3, 3]
        target_pose[3:] = R.from_matrix(pose[:3, :3]).as_rotvec()
        target_pose = target_pose.tolist()
        rtde_c.moveL(target_pose, 0.05)
        print("Action executed")
    elif ans == 'e':
        print("Exiting...")
        exit(0)
    else:
        print("Action not executed")
