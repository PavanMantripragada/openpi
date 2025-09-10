import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

# Config and checkpoint of trained model
config = config.get_config("pi0_fast_ego_mimic_low_mem_finetune")
# checkpoint_dir = "checkpoints/pi0_fast_ego_mimic_low_mem_finetune/joint_actions_full_image_sparse/25000"
checkpoint_dir = "checkpoints/pi0_fast_ego_mimic_low_mem_finetune/delta_ee_poses_sparse/22000"

episode_id = 0
for episode_id in range(19,24):
    # Episode directory for rollout
    episode_dir = f"third_party/ego_mimic/data_sparse_ee_pose/episode_{episode_id:05d}"

    # Create a trained policy.
    policy = policy_config.create_trained_policy(config, checkpoint_dir)


    # Find all PNG files named image_0000x.png in the episode_dir
    image_files = sorted([f for f in os.listdir(episode_dir) if re.match(r"image_.*\.png", f)])
    state_files = sorted([f for f in os.listdir(episode_dir) if re.match(r"state_.*\.npy", f)])
    action_files = sorted([f for f in os.listdir(episode_dir) if re.match(r"action_.*\.npy", f)])
    with open(os.path.join(episode_dir, "instruction.txt"), 'r') as file:
        instruction = file.read()
    actions = []
    pred_actions = []
    states = []
    pred_action_chunks = []
    for image_file, state_file, action_file in zip(image_files, state_files, action_files):
        image = cv2.imread(os.path.join(episode_dir, image_file))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        state = np.load(os.path.join(episode_dir, state_file))
        action = np.load(os.path.join(episode_dir, action_file))
        states.append(state)
        actions.append(action)
        # Run inference on a dummy example.
        example = {
            "observation/state": state,
            "observation/image": image,
            "observation/wrist_image": image,
            "prompt": instruction,
        }


        action_chunk = policy.infer(example)["actions"]
        print(f"action chunk shape: {action_chunk.shape}")
        pred_actions.append(action_chunk[0])
        pred_action_chunks.append(action_chunk)
        # print(type(action_chunk), action_chunk.shape, action_chunk)
        # print(action)
        # print("donnnknkn")
        # Display the image using matplotlib
        # plt.imshow(image)
        # plt.show()
        # Create a named window with the option to resize
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.imshow("image", image)
        # # Resize the window (width, height)
        # # cv2.resizeWindow("image", 800, 600)
        # print("donnnknkn")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    states = np.array(states)
    actions = np.array(actions)
    pred_actions = np.array(pred_actions)
    pred_action_chunks = np.array(pred_action_chunks)
    
    output_dir = "data/sparse_model"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"sparse_episode_{episode_id:05d}_states.npy"), states)
    np.save(os.path.join(output_dir, f"sparse_episode_{episode_id:05d}_actions.npy"),actions)
    np.save(os.path.join(output_dir, f"sparse_episode_{episode_id:05d}_pred_actions.npy"), pred_actions)
    np.save(os.path.join(output_dir, f"sparse_episode_{episode_id:05d}_pred_action_chunks.npy"), pred_action_chunks)

    error = 180*np.abs(actions - pred_actions)/np.pi
    print("Mean Error (degrees):", np.mean(error, axis=0))
    print("Max Error (degrees):", np.max(error, axis=0))
    print("Min Error (degrees):", np.min(error, axis=0))
    print("Standard Error (degrees):", np.min(error, axis=0))



    fig, axs = plt.subplots(7, 1, figsize=(10, 20))
    for i in range(7):
        axs[i].plot(180*actions[:, i]/np.pi, label='Actual Actions', linestyle='solid',marker='x')
        axs[i].plot(180*pred_actions[:, i]/np.pi, label='Predicted Actions', linestyle='dashed',marker='x')
        axs[i].plot(180*states[:, i]/np.pi, label='State', linestyle='solid', alpha=0.5, color='red',marker='x')
        axs[i].set_title(f'Joint {i+1}')
        axs[i].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sparse_episode_{episode_id:05d}_traj.png"))
    plt.show()

    fig, axs = plt.subplots(7, 1, figsize=(10, 20))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(7):
        axs[i].hist(error[:, i], bins=20, color=colors[i])
        axs[i].set_title(f'Error Distribution for Joint {i+1}')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sparse_episode_{episode_id:05d}_hist.png"))
    plt.show()

