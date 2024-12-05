import math
from argparse import ArgumentParser

import numpy
import numpy as np
import torch

from helpers import *
from style_utils import style_layer_names


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default='s2_wild512')
    parser.add_argument('--output_dir', type=str, default='styles')

    parser.add_argument('--n_codes', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def get_activation(name, container):
    def hook(model, input, output):
        container[name] = output.detach().cpu()

    return hook


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path, n_layer = parse_generator_fp(args.domain)
    model = prepare_model(args.domain, model_path, device)

    # Attach hooks for alternative model
    affine_transforms = {}
    layer_names = style_layer_names(args.domain)
    for layer_name_cont in layer_names:
        module = model.synthesis
        for layer_n in layer_name_cont:
            module = getattr(module, f'{layer_n}')
        module.affine.register_forward_hook(get_activation(''.join(layer_name_cont), affine_transforms))

    # Prepare latent codes.
    torch.manual_seed(0)
    z_codes = torch.randn(size=(args.n_codes, 512)).to(device)

    # Prepare container
    print(layer_names)
    for i, ln in enumerate(layer_names):
        print(i, ": ", ln)

    affine_transform_container = {''.join(key): [] for key in layer_names}
    mean_container, std_container = affine_transform_container.copy(), affine_transform_container.copy()

    w_codes = []
    for z_idx, z_code in enumerate(z_codes):
        print(z_idx, ' / ', len(z_codes))
        image, ws = model.forward(z_code.unsqueeze(0),
                                  None,
                                  truncation_psi=0.7,
                                  truncation_cutoff=8,
                                  noise_mode='const')
        w_codes.append(ws.detach().cpu())

        for k, v in affine_transforms.items():
            affine_transform_container[k].append(affine_transforms[k].clone())

    # Clean up container
    for k, v in affine_transform_container.items():
        new_tensor = torch.stack(affine_transform_container[k]).squeeze()
        affine_transform_container[k] = new_tensor

    # Compute mean and std
    for k, v in affine_transform_container.items():
        styles = affine_transform_container[k]  # [n_code, n_style_dim]
        mean, std = torch.mean(styles), torch.std(styles)
        mean_container[k] = mean
        std_container[k] = std

    # Save
    output_directory = f'{args.output_dir}'
    w_fp = os.path.join(output_directory, f'{args.domain}-w.pt')
    style_fp = os.path.join(output_directory, f'{args.domain}-styles.pt')
    style_mean_fp = os.path.join(output_directory, f'{args.domain}-styles-mean.pt')
    style_std_fp = os.path.join(output_directory, f'{args.domain}-styles-std.pt')
    os.makedirs(output_directory, exist_ok=True)

    torch.save(w_codes, w_fp)
    torch.save(affine_transform_container, style_fp)
    torch.save(mean_container, style_mean_fp)
    torch.save(std_container, style_std_fp)

