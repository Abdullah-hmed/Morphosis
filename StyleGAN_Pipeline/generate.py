import os
import sys
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import base64
from PIL import Image
# Dynamically determine the root directory based on the location of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(os.path.join(current_dir, "stylegan2_ada_pytorch"))
print('generator directory', current_dir)
from legacy_loader import load_network_pkl # type: ignore (Path is defined using currentdir hence suppressing error)
import dnnlib # type: ignore (Path is defined using currentdir hence suppressing error)

import torch

class ImageGenerator:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device: ",self.device)

    def synthesize_image(self, latent):
        # Load the pretrained weights ffhq.pkl
        network_pkl = os.path.join(current_dir, 'stylegan2_ada_pytorch', 'pretrained-weights', 'ffhq.pkl')
        
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)  # add batch dim
        elif latent.ndim == 1:
            latent = latent.unsqueeze(0).unsqueeze(0)
        elif latent.ndim > 3:
            latent = latent.squeeze(0)

        # Create the generator
        with dnnlib.util.open_url(network_pkl) as f:
            G = load_network_pkl(f)['G_ema'].to(self.device)
        # Generate Image
        synth_image = G.synthesis(latent, noise_mode='const')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        return synth_image
        
    def display_image(self, image_array, size=(512, 512)):
            root = tk.Tk()
            ImageViewer(root, image_array, size)
            root.mainloop()
            root.destroy()




    def generate_image(self, latent_vector):
        """
        Generates an image from a given latent vector and returns it as a base64 encoded PNG string.

        Args:
            latent_vector (torch.Tensor): The latent vector representing the image to be generated.
            
        Returns:
            str: A base64 encoded string of the generated image.
        """
        image_array = self.synthesize_image(latent_vector)
        # self.display_image(image_array)
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

class ImageViewer:
    def __init__(self, master, image_array, size):
        self.master = master
        self.master.title("Generated Image")
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        
        # Resize for better viewing
        image = image.resize(size, Image.Resampling.LANCZOS)

        # Convert PIL image to PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Create a label and add the image
        self.label = tk.Label(master, image=photo)
        self.label.image = photo  # Keep a reference!
        self.label.pack()

        # Add a button to close the window
        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.pack()