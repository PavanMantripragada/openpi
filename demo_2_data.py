import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def get_relative_poses(future_pose,current_pose):

    """
    Calculate the relative poses between future and current poses using rotation vectors.

    Args:
        future_pose (np.ndarray): Future pose in GE3.
        current_pose (np.ndarray): Current pose in GE3.

    Returns:
        np.ndarray: Relative pose in the format [dx, dy, dz, rx, ry, rz].
    """
    # Extract positions and orientations
    pos_future = future_pose[:3,3].reshape(3)
    pos_current = current_pose[:3,3].reshape(3)

    # Calculate relative position
    relative_position = pos_future - pos_current

    # Calculate relative orientation using rotation vectors
    r_future = R.from_matrix(future_pose[:3,:3])
    r_current = R.from_matrix(current_pose[:3,:3])
    relative_orientation = (r_future * r_current.inv()).as_rotvec()

    return np.concatenate((relative_position, relative_orientation))


action_type = "pouring"

episode_paths = [f"../Imitation/scripts/data/{action_type}/inlab_{action_type}_demos_{i}" for i in range(0,20)]


data_path = f"third_party/ego_mimic/data_joint_angles_{action_type}_train"

for episode_id, episode_path in enumerate(episode_paths):
    # Load the flange poses from the numpy file
    npy_file = f"{episode_path}/joint_angles.npy"
    if not os.path.exists(npy_file):
        print(f"File {npy_file} does not exist.")
        continue
    joint_angles = np.load(npy_file)

    # Create the episode directory
    episode_dir = f"{data_path}/episode_{episode_id:05d}"
    os.makedirs(episode_dir, exist_ok=True)

    for step in range(len(joint_angles)):
        
        # Copy the image file from the episode path to the episode directory
        img_path = f"{episode_path}/image_{step:05d}.png"
        img_file = f"{episode_dir}/image_{step:05d}.png"
        os.system(f"cp {img_path} {img_file}")

        # Save the flange pose as a numpy file
        state_file = f"{episode_dir}/state_{step:05d}.npy"
        state = np.append(joint_angles[step],1.0)
        np.save(state_file, state)

        # Save the flange pose as a numpy file
        action_file = f"{episode_dir}/action_{step:05d}.npy"
        if step < len(joint_angles) - 1:
            action = np.append(joint_angles[step + 1], 1.0)
        else:
            action = np.append(joint_angles[step], 0.0)
        np.save(action_file, action)
    # for step in range(0, len(joint_angles), 10):
        
    #     # Copy the image file from the episode path to the episode directory
    #     img_path = f"{episode_path}/image_{step:05d}.png"
    #     img_file = f"{episode_dir}/image_{step:05d}.png"
    #     os.system(f"cp {img_path} {img_file}")

    #     # Save the state as a numpy file
    #     state_file = f"{episode_dir}/state_{step:05d}.npy"
    #     state = np.append(np.concatenate((joint_angles[step][:3,3].reshape(3), R.from_matrix(joint_angles[step][:3,:3]).as_rotvec())), 1.0)
    #     np.save(state_file, state)

    #     # Save the action as a numpy file
    #     action_file = f"{episode_dir}/action_{step:05d}.npy"
    #     next_step = step + 10
    #     if next_step < len(joint_angles):
    #         print("-----------------------",joint_angles[next_step])
    #         action = np.append(get_relative_poses(joint_angles[next_step],joint_angles[step]), 1.0)
    #         print("@@@@@@@@@@",get_relative_poses(joint_angles[next_step],joint_angles[step]))
    #     else:
    #         action = np.append(np.concatenate((joint_angles[step][:3,3].reshape(3), R.from_matrix(joint_angles[step][:3,:3]).as_rotvec())), 0.0)
    #         print(np.concatenate((joint_angles[step][:3,3].reshape(3), R.from_matrix(joint_angles[step][:3,:3]).as_rotvec())))
    #     np.save(action_file, action)

    languation_instruction = action_type
    with open(os.path.join(episode_dir, f"instruction.txt"), 'w') as file:
        file.write(languation_instruction)

