import os
import sys
import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
class LatentOperation:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.effect_latents_dict = {}
        self.load_effect_latents()
    def load_effect_latents(self):
        effect_latents_directory = os.path.join(current_dir, 'effect_latents')
        
        for filename in os.listdir(effect_latents_directory):
            if filename.endswith('.npy'):
                # Load name of file
                name = os.path.splitext(filename)[0]

                # Load latent vector
                latent_path = os.path.join(effect_latents_directory, filename)
                latent = np.load(latent_path)
                latent = torch.tensor(latent, device=self.device).unsqueeze(0)

                self.effect_latents_dict[name] = latent
        else:
            print(f'Skipping loading {filename}')

        for name, array in self.effect_latents_dict.items():
            print(f"Array '{name} shape: {array.shape}")
    
    def perform_edit(self, base_latent: torch.Tensor, effect_dict: dict):
        """
        Applies the specified effects to the base latent tensor.

        Parameters:
            base_latent (torch.Tensor): The base latent tensor to which the effects will be applied.
            effect_dict (dict): A dictionary containing the effects to be applied, where the keys are the names of the effects and the values are the strengths of the effects.

        Returns:
            torch.Tensor: The edited latent tensor after applying the effects.

        """
        result_latent = torch.zeros_like(base_latent)

        for effect, strength in effect_dict.items():

            effect_latent = self.effect_latents_dict[effect]

            # strength = max(0.0, min(1.0, strength))

            
            # Apply the effect with the specified strength
            result_latent = result_latent + (effect_latent * (strength*0.5))
        edited_latent = base_latent + result_latent
        return edited_latent
