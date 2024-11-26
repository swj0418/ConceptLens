import torch
import natsort
import numpy as np

def prepare_walk_dataset_directory(domain, method, layer):
    """
    Prepares walk dataset directory structures

    Args:
        experiment_name:

    Returns:

    """

    output_root = output_root = f''


def get_layer_permutation(n_layers=16, min_layer_count=2, max_layer_count=5, layer_range: list = None):
    assert len(layer_range) == 2

    size = np.random.randint(min_layer_count, max_layer_count, dtype=int)
    if layer_range:
        layers = np.random.choice(range(layer_range[0], layer_range[1]), size, replace=False)
    else:
        layers = np.random.choice(range(n_layers), size, replace=False)

    layers = layers.tolist()
    layers = natsort.natsorted(layers)
    return layers


def get_onehot_layer_permutation(n_layers=16, min_layer_count=2, max_layer_count=5, layer_range: list = None):
    layers = get_layer_permutation(n_layers, min_layer_count, max_layer_count, layer_range)
    onehot = [1 if i in layers else 0 for i in range(n_layers)]
    return onehot, layers


def pca_direction(code_collection, top_dir, skip_first=True):
    # SVD
    # Dimension per row.
    # U, S, Vh = torch.linalg.svd(code_collection, full_matrices=False)
    # if top_dir is not None:
    #     offset = 2 if skip_first else 1
    #     eigvec = Vh[offset - 1: top_dir + (offset - 1), :].T

    # PCA
    mean_adj_w = code_collection - torch.mean(code_collection, dim=0, keepdim=True)  # [code, dim]
    w_cov = mean_adj_w.T @ mean_adj_w
    eigval, eigvec = np.linalg.eigh(w_cov.detach().cpu().numpy())
    eigvec = torch.tensor(eigvec).T
    eigvec = torch.flip(eigvec, dims=[0])
    # print(code_collection.shape)

    # estimator = PCAEstimator(code_collection.shape[1])
    # estimator.fit(code_collection)
    # eigvec, stdev, var_ratio = estimator.get_components()  # Row is direction

    if top_dir is not None:
        offset = 2 if skip_first else 1
        # eigvec = eigvec[:, -top_dir - offset:-offset]
        eigvec = eigvec[:top_dir, :]

    return torch.tensor(eigvec, dtype=torch.float)


def pca_direction_subsample(code_collection, top_dir, subsample_size, skip_first=True):
    # SVD
    # Dimension per row.
    # U, S, Vh = torch.linalg.svd(code_collection, full_matrices=False)
    # if top_dir is not None:
    #     offset = 2 if skip_first else 1
    #     eigvec = Vh[offset - 1: top_dir + (offset - 1), :].T

    code_idx = torch.randint(0, len(code_collection), subsample_size)
    styles = code_collection[code_idx]

    # PCA
    mean_adj_w = styles - torch.mean(styles, dim=0, keepdim=True)  # [code, dim]
    w_cov = mean_adj_w.T @ mean_adj_w
    eigval, eigvec = np.linalg.eigh(w_cov.detach().cpu().numpy())
    eigvec = torch.tensor(eigvec)

    if top_dir is not None:
        offset = 2 if skip_first else 1
        eigvec = eigvec[:, -top_dir - offset:-offset]

    return eigvec
