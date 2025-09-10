"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import os
from pathlib import Path
import re
import cv2
import numpy as np

REPO_NAME = "mppavan/ego_mimic"  # Name of the output dataset, also used for the Hugging Face Hub

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ur10",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (1242, 2208, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (1242, 2208, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    episode_dirs = sorted(
        [
            Path(data_dir) / d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("episode_")
        ]
    )
    for episode_dir in episode_dirs:
        # Find all PNG files named image_0000x.png in the episode_dir
        image_files = sorted([f for f in os.listdir(episode_dir) if re.match(r"image_.*\.png", f)])
        state_files = sorted([f for f in os.listdir(episode_dir) if re.match(r"state_.*\.npy", f)])
        action_files = sorted([f for f in os.listdir(episode_dir) if re.match(r"action_.*\.npy", f)])
        for image_file, state_file, action_file in zip(image_files, state_files, action_files):
            image = cv2.imread(os.path.join(episode_dir, image_file))
            state = np.load(os.path.join(episode_dir, state_file))
            action = np.load(os.path.join(episode_dir, action_file))

            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": np.zeros_like(image),
                    "state": state,
                    "actions": action,
                }
            )
        with open(os.path.join(episode_dir, "instruction.txt"), 'r') as file:
            instruction = file.read()
        dataset.save_episode(task=instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)
