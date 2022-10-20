import os
import argparse

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models import EmptierGAN

from PIL import Image
from tqdm import tqdm
import glob as glob


resolution = 256
stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
transform = transforms.Compose([
    t for t in [
        transforms.Resize([resolution, resolution*2], InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ]
])
normalize_transform = transforms.Normalize(stats['mean'], stats['std'], inplace=True)


def load_model(pretrain_weights_path, device='cuda'):
    print("Model loading...")
    ckpt = torch.load(pretrain_weights_path, device)
    for key in ckpt['state_dict'].copy():
        key_prefix = key.split('.')[0]
        if key_prefix == "generator_f2e":
            new_key = key.split(key_prefix + '.')[1]
            ckpt['state_dict'][new_key] = ckpt['state_dict'].pop(key)
        else:
            ckpt['state_dict'].pop(key)

    model = EmptierGAN().to(device)
    model.load_state_dict(state_dict=ckpt['state_dict'])
    print("Model loaded!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_path", default='data',
                        help='path to your dataset')
    parser.add_argument("--save_path", default='output',
                        help='path to save the output images')
    parser.add_argument("--pretrain_weights_path", default='pretrain_weights/v2_256x512_epoch_644-step_210749.ckpt',
                        help="path to pretrained ckpt file path")
    parser.add_argument('--device', default='cuda',
                        help='Specify the device on which to run the code, in PyTorch syntax, '
                             'e.g. `cuda`, `cpu`, `cuda:3`.')
    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    torch.set_grad_enabled(False)
    model = load_model(args.pretrain_weights_path, args.device)

    image_paths = glob.glob(os.path.join(args.dataset_path, '*'))
    image_paths.sort()

    for i in tqdm(range(len(image_paths)), desc='Running for images'):
        input_path = image_paths[i]
        if '/' in input_path:
            file_name = input_path.split('/')[-1].split('.')[0]
        elif '\\' in input_path:
            file_name = input_path.split('\\')[-1].split('.')[0]
        else:
            raise NotImplementedError

        input = normalize_transform(torch.unsqueeze(transform(Image.open(input_path)), 0)[:, 0:3, :, :]).to(args.device)

        out = model(input)
        out = ((out.detach()[0].permute(1, 2, 0) + 1.) / 2. * 255.).clamp(min=0, max=255).cpu().numpy().astype(np.uint8)

        Image.fromarray(out).save(os.path.join(save_path, file_name + '_output' + '.png'))