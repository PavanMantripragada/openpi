from PIL import Image
import os
import numpy as np

def create_blank_image(file_path):
    image = Image.new('RGB', (256, 256), (255, 255, 255))
    image.save(file_path, 'PNG')

if __name__ == "__main__":

    dummy_data_dir = "third_party/ego_mimic/data"
    for i in range(300):
        episode_dir = os.path.join(dummy_data_dir, f"episode_{i:05d}")
        os.makedirs(episode_dir)
        for j in range(150):
            create_blank_image(os.path.join(episode_dir, f"image_{j:05d}.png"))
            action = np.random.rand(7).astype(np.float32)
            state = np.random.rand(7).astype(np.float32)
            np.save(os.path.join(episode_dir, f"action_{j:05d}.npy"), action)
            np.save(os.path.join(episode_dir, f"state_{j:05d}.npy"), state)
            languation_instruction = "open"
        with open(os.path.join(episode_dir, f"instruction.txt"), 'w') as file:
            file.write(languation_instruction)