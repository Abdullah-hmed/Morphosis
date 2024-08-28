import json
import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

# Dynamically determine the root directory based on the location of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'outputs',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'outputs')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location=device)
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.to(device)  # Ensure the model is on the GPU


    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.to(device).float()
            tic = time.time()
            result_batch, latents_batch = run_on_batch(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]
            folder_name = os.path.splitext(os.path.basename(im_path))[0]
            # Save latent vector
            latent = latents_batch[i].cpu().numpy()
            latent_filename = folder_name + ".npy"
            latent_path = os.path.join(out_path_results, folder_name, latent_filename)
            np.save(latent_path, latent)

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                        np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                        np.array(result.resize(resize_amount))], axis=1)
                else:
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                        np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, folder_name, os.path.basename(im_path))
            print("Image Save Path after Inference:", im_save_path)
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)
    
    print(latent)



def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch, latents = net(inputs, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        latents = []
        for image_idx, input_image in enumerate(inputs):
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            res, latent = net(input_image.unsqueeze(0).to("cuda").float(),
                              latent_mask=latent_mask,
                              inject_latent=latent_to_inject,
                              alpha=opts.mix_alpha,
                              resize=opts.resize_outputs,
                              return_latents=True)
            result_batch.append(res)
            latents.append(latent)
        result_batch = torch.cat(result_batch, dim=0)
        latents = torch.cat(latents, dim=0)
    return result_batch, latents


if __name__ == '__main__':
    run()
