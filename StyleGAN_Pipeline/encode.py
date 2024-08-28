import json
import subprocess
import torch
import shutil
import os
from os import path
import sys
import numpy as np
from PIL import Image
from face_crop_plus import Cropper
from tkinter import * 
from tkinter.filedialog import askopenfilename

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print('encoder directory: ',current_dir)

from pixel2style2pixel.scripts import inference

class NoFaceDetectedException(Exception):
    print("No face detected in image")

class ImageEncoder:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.outputs_dir = os.path.join(self.script_dir, 'experiment/outputs')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_inference(self,
                    folder_name,
                    inference_path="pixel2style2pixel/scripts/inference.py",
                    experiment_directory="experiment",
                    checkpoint_path="experiment/checkpoint/psp_ffhq_encode.pt",
                    data_path="experiment/outputs",
                    test_batch_size = 1,
                    test_workers=4
                    ):
        inference_path = os.path.join(self.script_dir, inference_path)
        experiment_directory = os.path.join(self.script_dir, experiment_directory)
        checkpoint_path = os.path.join(self.script_dir, checkpoint_path)
        data_path = os.path.join(self.script_dir, data_path, folder_name)

        command = (
            f"python3 {inference_path} "
            f"--exp_dir {experiment_directory} "
            f"--checkpoint_path {checkpoint_path} "
            f"--data_path {data_path} "
            f"--test_batch_size {test_batch_size} "
            f"--test_workers {test_workers} "
            f"--couple_outputs"
        )
        print(command)
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Output:\n", result.stdout)
        print("Error:\n", result.stderr)



    def MoveFile(self, file_path):
        if file_path:
            print(f"Selected file: {file_path}")

            file_name = os.path.basename(file_path)
            latent = os.path.splitext(file_name)[0]+".npy"
            folder_name = os.path.splitext(file_name)[0]
            # Where to move the selected image
            destination_folder = os.path.join(self.script_dir, "experiment/data", folder_name)
            
            # if destination folder doesn't exist, make one
            if not os.path.exists(destination_folder):
                print("no folder, making one.")
                os.makedirs(destination_folder)
            
            


            destination_path = os.path.join(destination_folder, file_name)
            
            # Copying the file to destination folder
            shutil.copy(file_path, destination_path)
            print(f"File moved to: {destination_path}, destination folder: {destination_folder}")
            print(file_name, latent)
            return folder_name, file_name, latent
        else:
            print("No file selected")
            return None

    def process_image(self, folder_name):
        # Make use of face_crop_plus to detect and align faces
        cropper = Cropper(face_factor=0.7, strategy="largest", padding="replicate")
        input_dir = os.path.join(self.script_dir, "experiment/data", folder_name)
        output_dir = os.path.join(self.script_dir, "experiment/outputs", folder_name)
        # Perform processing and output images to 'Outputs'
        cropper.process_dir(input_dir=input_dir, output_dir=output_dir)
        if not os.path.exists(output_dir):
            raise NoFaceDetectedException("No face detected in the uploaded image.")

    def load_latents(self, folder_name, latent):
        latent = np.load(os.path.join(self.outputs_dir, folder_name, latent))
        latent = torch.tensor(latent, device=self.device).unsqueeze(0)
        return latent

    def encode_image(self, image_path: str):
        """
        Encodes an image into a latent representation.

        Parameters:
        image_path (str): The path to the image to be encoded.

        Returns:
        latent (torch.tensor): The latent representation of the encoded image.
        """
        folder_name, image_name, latent_name = self.MoveFile(image_path)
        self.process_image(folder_name)
        self.run_inference(folder_name)
        latent = self.load_latents(folder_name, latent_name)
        return latent