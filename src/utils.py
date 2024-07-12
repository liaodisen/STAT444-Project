import random
import numpy as np
import torch

def seed_everything(seed: int):
    """
    Seed all random number generators to ensure reproducibility.

    Parameters:
    seed (int): The seed to use for the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

