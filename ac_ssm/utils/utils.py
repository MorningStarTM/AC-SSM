import numpy as np
import os
from ac_ssm.utils.logger import logger

def save_episode_rewards(rewards, save_dir, filename="episode_rewards.npy"):
    """
    Saves the episode rewards as a .npy file.

    Args:
        rewards (list or np.ndarray): List of episode rewards to save.
        save_dir (str): Directory path where the file will be saved.
        filename (str): Name of the file to save the rewards in. Default is 'episode_rewards.npy'.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, np.array(rewards))
    logger.info(f"Episode rewards saved to {save_path}")


def load_episode_rewards(file_path):
    """
    Loads the episode rewards from a .npy file.

    Args:
        file_path (str): Full path to the .npy file containing saved episode rewards.

    Returns:
        np.ndarray: Loaded array of episode rewards.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    rewards = np.load(file_path)
    logger.info(f"Loaded episode rewards from {file_path}")
    return rewards
