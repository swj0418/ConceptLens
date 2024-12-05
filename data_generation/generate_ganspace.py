import os
import pickle
import re
from argparse import ArgumentParser

from style_utils import read_styles
from model_utils import *

from sklearn.cluster import KMeans
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from ganspace_utils import direction_lstsq
from image_utils import convert_images_to_uint8
from latent_code_dataset import LatentCodeDataset
from stylegan2 import Generator as G2
from stylegan3 import Generator as G3
from walk_utils import pca_direction

from helpers import parse_layer_configuration, parse_generator_fp, prepare_model, ConceptLensDataset


def read_affine_w(affine_n_code=None, path='ffhq-affvec.pt'):
    affine_transforms = torch.load(path)

    all_affine_w = []
    layer_keys = affine_transforms[0].keys()
    for layer_key in layer_keys:
        tmp = []
        if affine_n_code:
            for code_i in range(affine_n_code):
                code = affine_transforms[code_i]
                item = code[layer_key]
                tmp.append(item)
        else:
            for code in affine_transforms:
                item = code[layer_key]
                tmp.append(item)

        layer_affine_w = torch.stack(tmp).squeeze()
        all_affine_w.append(layer_affine_w)
    return all_affine_w  # I cannot stack this list because dimensions vary from layers to layers.


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--layer')
    parser.add_argument('--method', default='ganspacekmc')
    parser.add_argument('--application', type=str, default='layerwise')
    parser.add_argument('--exp_name', type=str, default=None)

    parser.add_argument('--n_codes', type=int, default=400)
    parser.add_argument('--top_dir', type=int, default=18)
    parser.add_argument('--edit_dist', type=int, default=5)
    parser.add_argument('--n_perm', type=int, default=8)
    parser.add_argument('--batchsize', type=int, default=20)

    parser.add_argument('--n_affine', type=int, default=None, help="Number of latent codes that are used in PCA.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pseed', type=int, default=18, help="Seed used for sampling layer combinations.")

    return parser.parse_args()


# def pca_direction(code_collection, top_dir, skip_first):
#     # Calculate mean and center the data
#     mean = torch.mean(code_collection, dim=0)
#     centered_data = code_collection - mean
#
#     # Calculate covariance matrix
#     covariance_matrix = torch.matmul(centered_data.T, centered_data) / (centered_data.shape[0] - 1)
#
#     # Eigenvalue decomposition
#     eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
#
#     # Sort the eigenvectors based on eigenvalues in descending order
#     sorted_indices = torch.argsort(eigenvalues, descending=True)
#     sorted_eigenvectors = eigenvectors[:, sorted_indices]
#
#     # Skip the first eigenvector if skip_first is True
#     start_index = 1 if skip_first else 0
#
#     # Select the top 'top_dir' eigenvectors
#     top_eigenvectors = sorted_eigenvectors[:, start_index:start_index + top_dir]
#
#     return top_eigenvectors.T


# Example usage
# code_collection = torch.randn(100, 50) # Example data
# top_dir = 5
# skip_first = True
# top_directions = pca_direction(code_collection, top_dir, skip_first)


if __name__ == '__main__':
    args = parse_args()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # SeFA sampling hyper-parameters
    sampling_rate = 16
    sub_minimum, sub_maximum = 32, 65536
    top_direction_per_sample = 6
    n_direction = 30
    skip_first_svd = True

    model_path, generator_size = parse_generator_fp(args.domain)
    layer_range = parse_layer_configuration(args.layer, generator_size)
    print("LAYER RANGE: ", layer_range)
    layer_range = [_ for _ in range(12)]

    concept_lens_data = ConceptLensDataset(args.domain, args.application, args.layer, args.method, args.exp_name)

    # Get generator
    generator = prepare_model(domain=args.domain, model_path=model_path, device=device)

    torch.manual_seed(args.seed)
    generator = generator.to(device)
    z_codes = torch.randn(size=(args.n_codes, 512)).to(device)
    all_affine_w = read_styles(domain=args.domain,
                               output_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             'styles'))
    all_affine_w = list(all_affine_w.values())

    # Gather linear weights
    # weights = {k: v for k, v in generator.named_parameters() if
    #            re.match("synthesis\.b[0-9]+\.[A-Za-z]+[0-1]+\.affine\.weight", k)}
    # weights = list(weights.values())

    weights = {k: v.detach().cpu() for k, v in generator.named_parameters() if
               re.match("synthesis\.+(.*)+(conv.*)+\.affine\.weight", k)}

    if args.domain == 's3_landscape256':
        weights = {k: v.detach().cpu() for k, v in generator.named_parameters() if
                   re.match("synthesis\.+(.*)+\.affine\.weight", k)}
    print(weights.keys())
    # weights = list(weights.values())[1:] # Why skip???
    weights = list(weights.values())

    # biases = {k: v for k, v in generator.named_parameters() if
    #            re.match("synthesis\.b[0-9]+\.[A-Za-z]+[0-1]+\.affine\.bias", k)}
    # biases = list(biases.values())

    layer_dimensionality = [we.shape[0] for we in weights]

    # Compute directions - for all layer combinations.
    style_directions = []
    norms = []
    w_directions = []
    for trial in range(sampling_rate):
        print(f"Generating Directions: {trial}/{sampling_rate}")
        target_affine = torch.cat([all_affine_w[layer_idx] for layer_idx in layer_range], dim=-1)

        # Subsample
        sample_idx = torch.randint(0, len(target_affine), size=(np.random.randint(sub_minimum, sub_maximum),))
        styles = target_affine[sample_idx]

        # unit norm, [direction, dimension]
        affine_eigvec = pca_direction(styles, top_dir=top_direction_per_sample).to(device)  # Return in row direction.
        # affine_eigvec = pca_direction(styles, top_dir=top_direction_per_sample, skip_first=True).to(device)

        print(affine_eigvec.shape)

        # print(trial, styles.shape)
        style_directions.append(affine_eigvec)

    # [many, style dim]
    style_directions = torch.cat(style_directions).detach().cpu().numpy()

    # Run K means
    cluster = KMeans(n_clusters=n_direction, n_init="auto").fit(style_directions)
    style_directions = torch.tensor(style_directions)
    # directions = torch.tensor(cluster.cluster_centers_).to(device).to(torch.float)
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    # Find the closest direction to each direction cluster
    closest_directions = []
    for cluster_center in torch.tensor(cluster.cluster_centers_):
        normalized_cluster_center = cluster_center / torch.norm(cluster_center)
        sims = torch.cosine_similarity(normalized_cluster_center.unsqueeze(0), style_directions)
        closest_direction = torch.argmax(sims)
        closest_directions.append(style_directions[closest_direction])

    directions = torch.stack(closest_directions).to(device)

    # Project back to W space
    # [concat(output), input]
    perm_weights = torch.cat([weights[layer_idx] for layer_idx in layer_range], dim=0)
    # perm_biases = torch.cat([biases[layer_idx] for layer_idx in layer_range], dim=0)

    """
    Least square approximation of direction in the style space to W space.

    Bias vectors are subtracted from the direction vectors in the style space for better approximation.
    """
    # [M, N] / [M, K] => [concat(output), input] / [concat(output), n direction]
    print(directions.shape, perm_weights.shape)  # [192, 1024], []
    V, res, rank, S = direction_lstsq(directions, perm_weights)
    print(f"Mean Error: {np.mean(res)}, Mean Std: {np.std(res)}")

    directions = torch.tensor(V.T)

    # Normalize
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    directions = directions.to(device)

    torch.manual_seed(args.seed)
    z_codes = torch.randn(size=(args.n_codes, 512)).to(device)
    code_dataset = LatentCodeDataset(z_codes)
    dataloader = DataLoader(code_dataset, batch_size=args.batchsize, shuffle=False)

    # Generate codes
    wss = []
    for zi, z in enumerate(z_codes):
        with torch.no_grad():
            img, ws = generator.forward(z.unsqueeze(0), None, 0.7, 8, noise_mode='const')
            wss.append(ws)
        img = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)[0]
        fp = os.path.join(concept_lens_data.get_code_output_root(), f'{zi}.jpg')
        Image.fromarray(img).save(fp)

    # averaged standard deviation of dimensions in W space.
    wss = torch.stack(wss)  # [ncode, nlayer, wdim]
    wss = wss.squeeze(1)

    # averaged standard deviation of dimensions in W space.
    # wss = torch.stack(wss)  # [n_code, n_layer, w_dim]
    ws_std = torch.mean(torch.std(wss[:, 0], dim=0))
    ws_edit_dist = args.edit_dist * ws_std  # Unit
    edit_dist = args.edit_dist  # Unit
    print(f"Edit Distance: {edit_dist}, Relative edit distance to average std: {ws_edit_dist}")

    ind_dim_std = torch.std(wss[:, 0], dim=0)
    # print(ind_dim_std.shape)
    # print(ind_dim_std)

    # Generate data
    pbar = tqdm.tqdm(dataloader)
    for batch_idx, code_batch in enumerate(pbar):
        for di, direction in enumerate(directions):
            # direction = direction / torch.norm(direction)
            # direction = direction * ind_dim_std * edit_dist

            layers_to_walk = layer_range
            if args.application == 'global':
                layers_to_walk = None

            with torch.no_grad():
                img, ws = generator.forward_walking(code_batch, None, direction, edit_dist, layers_to_walk,
                                                    0.7, 8, noise_mode='const')

            walk_image = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)

            for img_idx, img in enumerate(walk_image):
                image_index = batch_idx * args.batchsize + img_idx
                fp = os.path.join(concept_lens_data.get_walked_output_root(), f'{image_index}-{di}.jpg')
                Image.fromarray(img).save(fp)

    pbar.close()

    # Save directions
    print(f"Saving directions => {directions.shape}")
    torch.save(directions.detach().cpu(), os.path.join(concept_lens_data.get_dataset_root(), 'directions.pt'))
    torch.save(wss.detach().cpu(), os.path.join(concept_lens_data.get_dataset_root(), 'codes.pt'))