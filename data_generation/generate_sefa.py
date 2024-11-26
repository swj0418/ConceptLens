import os
import random
import numpy as np
from argparse import ArgumentParser
from PIL import Image

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import tqdm

from latent_code_dataset import LatentCodeDataset
from walker import SeFAWalker, _compute_sefa_directions
from helpers import parse_layer_configuration, parse_generator_fp, prepare_model
from image_utils import convert_images_to_uint8


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default='ffhq')
    parser.add_argument('--layer', type=str)
    parser.add_argument('--method', type=str, default='sefakmc')
    parser.add_argument('--application', type=str, default='layerwise')
    parser.add_argument('--exp_name', type=str, default=None)

    parser.add_argument('--n_codes', type=int, default=400)
    parser.add_argument('--edit_dist', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=20)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--pseed', type=int, default=18, help="Seed used for sampling layer combinations.")

    return parser.parse_args()


def setup_directories(args):
    if not args.exp_name:
        output_root = f'output/{args.domain}-{args.method}-{args.application}-{args.layer}/'
    else:
        output_root = os.path.join('../data_walking/output', args.exp_name)

    code_image_output_root = os.path.join(output_root, 'codes')
    walked_image_output_root = os.path.join(output_root, 'walked')

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(code_image_output_root, exist_ok=True)
    os.makedirs(walked_image_output_root, exist_ok=True)

    return output_root, code_image_output_root, walked_image_output_root


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_directions(generator, layer_range, sampling_rate, top_direction_per_sample, skip_first_svd, device):
    style_weights = generator.get_style_weights()
    weight = torch.cat([layer_weight for i, layer_weight in enumerate(style_weights.values()) if i in layer_range])

    directions = []
    for trial in range(sampling_rate):
        drop_maximum = min(512, weight.shape[0])
        drop_minimum = max(64, drop_maximum - 1)
        drop_cut = np.random.randint(low=drop_minimum, high=drop_maximum)
        tmp = _compute_sefa_directions(style_weights, layers=layer_range, top_dir=top_direction_per_sample,
                                       skip_first=skip_first_svd, drop_cut=drop_cut).T
        directions.append(tmp)

    directions = torch.cat(directions)
    cluster = KMeans(n_clusters=192, n_init="auto").fit(directions)

    closest_directions = []
    for cluster_center in torch.tensor(cluster.cluster_centers_):
        normalized_cluster_center = cluster_center / torch.norm(cluster_center)
        sims = torch.cosine_similarity(normalized_cluster_center.unsqueeze(0), directions)
        closest_direction = torch.argmax(sims)
        closest_directions.append(directions[closest_direction])

    directions = torch.stack(closest_directions).to(device)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions


def generate_codes(generator, z_codes, output_root, code_image_output_root):
    wss = []
    for zi, z in enumerate(z_codes):
        with torch.no_grad():
            img, ws = generator.forward(z.unsqueeze(0), None, 0.7, 8, noise_mode='const')
        wss.append(ws)
        img = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)[0]
        fp = os.path.join(code_image_output_root, f'{zi}.jpg')
        Image.fromarray(img).save(fp)

    torch.save(torch.stack(wss).detach().cpu(), os.path.join(output_root, 'codes.pt'))
    return torch.stack(wss)


def main():
    args = parse_args()
    device = get_device()

    # SeFA sampling hyper-parameters
    sampling_rate = 1024
    top_direction_per_sample = 12
    skip_first_svd = True

    # Setup directories
    output_root, code_image_output_root, walked_image_output_root = setup_directories(args)
    print(f"Generating Dataset -----> {args.domain} {args.layer} ||| Output Root: {output_root}")

    # Prepare model
    model_path, n_layer = parse_generator_fp(args.domain)
    generator = prepare_model(domain=args.domain, model_path=model_path, device=device)
    layer_range = parse_layer_configuration(args.layer, n_layer)

    # Compute directions
    directions = compute_directions(generator, layer_range, sampling_rate, top_direction_per_sample, skip_first_svd, device)

    # Generate latent codes
    torch.manual_seed(args.seed)
    z_codes = torch.randn(size=(args.n_codes, 512)).to(device)
    code_dataset = LatentCodeDataset(z_codes)
    dataloader = DataLoader(code_dataset, batch_size=args.batchsize, shuffle=False)

    # Generate W codes and images
    wss = generate_codes(generator, z_codes, output_root, code_image_output_root)
    ws_std = torch.mean(torch.std(wss[:, 0], dim=0))
    ws_edit_dist = args.edit_dist * ws_std
    edit_dist = args.edit_dist
    print(f"Edit Distance: {edit_dist}, Relative edit distance to average std: {ws_edit_dist}")

    # Generate walked images
    pbar = tqdm.tqdm(dataloader)
    for batch_idx, code_batch in enumerate(pbar):
        for di, direction in enumerate(directions):
            layers_to_walk = None if args.application == 'global' else layer_range
            with torch.no_grad():
                img, _ = generator.forward_walking(code_batch, None, direction, edit_dist, layers_to_walk, 0.7, 8, noise_mode='const')
            walk_image = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)
            for img_idx, img in enumerate(walk_image):
                image_index = batch_idx * args.batchsize + img_idx
                fp = os.path.join(walked_image_output_root, f'{image_index}-{di}.jpg')
                Image.fromarray(img).save(fp)
    pbar.close()

    # Save directions
    print(f"Saving directions => {directions.shape}")
    torch.save(directions.detach().cpu(), os.path.join(output_root, 'directions.pt'))


if __name__ == '__main__':
    main()
