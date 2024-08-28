import os
import sys
import numpy as np
import torch
from encode import ImageEncoder
from generate import ImageGenerator
from latent_operations import LatentOperation

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
# Stack to keep track of the latest latent
latent_stack = [] 

# Temporary method
def load_latent_directly(latent_name: str):
    latent = np.load(os.path.join('experiment', 'outputs', latent_name))
    latent = torch.tensor(latent, device='cuda').unsqueeze(0)
    return latent


encoder = ImageEncoder()
generator = ImageGenerator()
modifier = LatentOperation()
# Calculate the latent vector of image and append it to stack
# latent = encoder.encode_image(image_path='experiment/sam.png') # TODO: Check to see why encoding process is slow (move to cuda)
latent = load_latent_directly("sam.npy")
latent_stack.append(latent)
generator.generate_image(latent_stack[-1]) 

# Performing edits
gender_latent = modifier.perform_edit(latent, "glasses", strength=3)
latent_stack.append(gender_latent)
generator.generate_image(latent_stack[-1])



