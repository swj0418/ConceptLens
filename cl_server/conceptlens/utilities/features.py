import scipy
import sklearn.metrics
import torch


def _subsetting(walk_features, original_features, direction_selection, code_selection):
    # Subset feature matrices
    if direction_selection:
        walk_features = walk_features[:, direction_selection]

    if code_selection:
        original_features = original_features[code_selection]
        walk_features = walk_features[code_selection]

    return walk_features, original_features


def _center(wf, of, c_mode, wm):
    if c_mode == 'individual_code_mean':  # Centering individual code.
        per_code_mean = torch.mean(wf, dim=1)
        centering_mean = per_code_mean.unsqueeze(1).repeat((1, wf.shape[1], 1))
        f = wf - centering_mean
    elif c_mode == 'all_code_mean':  # Centering on whole dataset.
        centering_mean = torch.mean(torch.mean(wf, dim=0), dim=0)
        f = wf - centering_mean
    elif c_mode == 'w_mean':
        f = wf - wm
    elif c_mode == 'global_mean':
        # Original + Walk embedding [code, direction, step, feature]
        expanded_of = of.unsqueeze(1).repeat(1, wf.shape[1], 1).unsqueeze(-2)
        all_embs = torch.cat([expanded_of, wf.unsqueeze(2)], dim=2)
        centering_mean = torch.mean(all_embs, dim=(0, 1, 2))
        f = wf - centering_mean
    else:
        f = wf
    return f


def _feature_centering(walk_features, original_features, code_selection, direction_selection, centering, prior_subset=False, w_mean=None):
    """
    Centers a feature shaped [code, direction, feature]. subset_code_mean requires subset gathering prior to centering,
    whereas other use-cases subset

    Args:
        centering: Centering option. 'individual' for centering per individual code. 'global' for centering all codes
        and directions with global mean.
        original_features: embedding of original images.
        code_selection: Index to select along code dimension. [] for no selection, [<int>] for selection
        direction_selection: Selection of directions indices.
        centering: centering methods
        prior_subset: controls whether to subset prior to centering.
        w_mean: mean of original embeddings.

    Returns:

    """
    assert centering in [None, 'individual_code_mean', 'all_code_mean', 'w_mean', 'global_mean']

    # print(centering, code_selection, direction_selection)
    if prior_subset:
        if direction_selection:
            walk_features = walk_features[:, direction_selection]

        if code_selection:
            original_features = original_features[code_selection]
            walk_features = walk_features[code_selection]
        features = _center(walk_features, original_features, c_mode=centering, wm=w_mean)
    else:
        features = _center(walk_features, original_features, c_mode=centering, wm=w_mean)
        if direction_selection:
            features = features[:, direction_selection]
        if code_selection:
            features = features[code_selection]

    # print("Feature Shape: ", features.shape)
    return features


def _feature_centering_2(walk_features, original_features, centering, w_mean=None):
    assert centering in [None, 'individual_code_mean', 'all_code_mean', 'w_mean', 'global_mean']
    return _center(walk_features, original_features, c_mode=centering, wm=w_mean)


def compute_mean_features(features):
    """
    Description here

    Args:
        experiment_names:

    Returns:

    """
    # This tensor is used to compute hierarchical clustering.
    centered_mean_features = torch.mean(features, dim=0)  # [direction x N, feature_dim] 360, 512

    # [n_code, n_dir, feat_dim]  [n_dir, feat_dim]
    return features, centered_mean_features


def compute_euclidean_distance(features, maintain_diagonal):
    """
    Computes code-wise cosine similarity of directions excluding self distance.

    Args:
        features:
        maintain_diagonal:

    Returns:

    """
    def _get_off_diagonal_elements(M):
        return M[~torch.eye(*M.shape, dtype=torch.bool)].view(M.shape[0], M.shape[1] - 1)

    # features = _feature_centering(features, centering, w_mean)

    if maintain_diagonal:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1)))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.euclidean_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            feature_cosine_distance[code_idx] = dist
    else:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1) - 1))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.euclidean_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            off_diagonal_dist = _get_off_diagonal_elements(dist)
            feature_cosine_distance[code_idx] = off_diagonal_dist

    return feature_cosine_distance


def compute_cosine_distance(features, maintain_diagonal):
    """
    Computes code-wise cosine similarity of directions excluding self distance.

    Args:
        features:
        maintain_diagonal:

    Returns:

    """
    def _get_off_diagonal_elements(M):
        return M[~torch.eye(*M.shape, dtype=torch.bool)].view(M.shape[0], M.shape[1] - 1)

    # features = _feature_centering(features, centering, w_mean)

    if maintain_diagonal:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1)))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.cosine_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            feature_cosine_distance[code_idx] = dist
    else:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1) - 1))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.cosine_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            off_diagonal_dist = _get_off_diagonal_elements(dist)
            feature_cosine_distance[code_idx] = off_diagonal_dist

    print(feature_cosine_distance.shape)
    return feature_cosine_distance


def postprocess_magnitude_features(walk_features,
                                  original_features,
                                  code_selection=None,
                                  direction_selection=None):
    """
    Magnitude features are always a diff vector without centering.

    Args:
        walk_features:
        original_features:
        mode:
        code_selection:
        direction_selection:
        centering:
        prior_subset:
        w_mean:
        normalize:

    Returns:

    """
    if direction_selection:
        walk_features = walk_features[:, direction_selection]

    if code_selection:
        walk_features = walk_features[code_selection]
        original_features = original_features[code_selection]

    features = walk_features - original_features.unsqueeze(1).repeat(1, walk_features.shape[1], 1)
    return features


def postprocess_clustering_features(walk_features,
                                    original_features,
                                    code_selection,
                                    mode,
                                    prior_subset,
                                    centering,
                                    w_mean):
    if prior_subset:
        walk_features, original_features = _subsetting(walk_features, original_features, None, code_selection)

    # Center
    walk_features = _feature_centering_2(walk_features, original_features, centering, w_mean)

    if not prior_subset:
        walk_features, _ = _subsetting(walk_features, original_features, None, code_selection)

    # Subtraction
    if mode == 'end':
        pass
    elif mode == 'diff':
        walk_features = walk_features - original_features.unsqueeze(1).repeat(1, walk_features.shape[1], 1)

    return walk_features


def postprocess_variance_features(walk_features,
                                  original_features,
                                  mode,
                                  code_selection=None,
                                  direction_selection=None,
                                  centering=None,
                                  prior_subset=False,
                                  w_mean=None,
                                  normalize=False):
    """

    Args:
        walk_features:
        original_features:
        mode:
        code_selection:
        direction_selection:
        centering:
        prior_subset:
        w_mean:
        normalize:

    Returns:

    """
    if prior_subset:
        walk_features, original_features = _subsetting(walk_features, original_features, direction_selection, code_selection)

    # Center
    walk_features = _feature_centering_2(walk_features, original_features, centering, w_mean)

    if not prior_subset:
        walk_features, _ = _subsetting(walk_features, original_features, direction_selection, code_selection)

    # Subtraction
    if mode == 'end':
        pass
    elif mode == 'diff':
        # print("Ori: ", original_features.shape, "   Walked: ", walk_features.shape)
        postprocessed_original_features = original_features.unsqueeze(1).repeat(1, walk_features.shape[1], 1)
        if code_selection:
            postprocessed_original_features = postprocessed_original_features[code_selection, :, :]
            walk_features = walk_features - postprocessed_original_features
        else:
            walk_features = walk_features - postprocessed_original_features

    if normalize:
        walk_features = walk_features / torch.norm(walk_features, dim=-1, keepdim=True)

    return walk_features


def compute_mean_features_pairwise_distances(features: torch.Tensor, dist_metric):
    """

    Args:
        features: [code, direction, feature]
        dist_metric: distance metric between feature vectors
        centering: center feature vectors.

    Returns:

    """

    # Centering the whole dataset - this is done in the feature post-processing.
    # if centering:
    #     # features = features - torch.mean(features, dim=(0, 1))
    #
    #     # Huh..?
    #     centering_mean = torch.mean(torch.mean(features, dim=0), dim=0)
    #     features = features - centering_mean
    #
    #     # So apparently, these two codes are different. The bottom one gives pdsit that is not nan while the
    #     # upper one does.. Why?

    # Compute pairwise distance of directions for each code.
    pairwise_distance = []
    for feat in features:
        d = scipy.spatial.distance.pdist(feat.numpy(), metric=dist_metric)  # I need a condensed form
        pairwise_distance.append(torch.tensor(d))
    pairwise_distance = torch.stack(pairwise_distance)

    mean_pairwise_distance = torch.mean(pairwise_distance, dim=0)  # Code average
    return mean_pairwise_distance


# def postprocess(walk_features,
#                 original_features,
#                 mode,
#                 code_selection=None,
#                 direction_selection=None,
#                 centering=None,
#                 prior_subset=False,
#                 w_mean=None,
#                 normalize=False):
#     """
#     Postprocess image embeddings of form [code, direction, feature dimension].
#     First filters selected code / direction.
#     Next, centers the data w.r.t. centering parameters.
#     Lastly, subtract or return the whole feature.
#
#     Args:
#         walk_features: embedding of walked images.
#         original_features: embedding of original images.
#         code_selection: Index to select along code dimension. [] for no selection, [<int>] for selection
#         direction_selection: Selection of directions indices.
#         mode: Whether to take a difference [diff], or the ending feature [end].
#         normalize: Row normalize feature tensors.
#         centering: centering methods
#         prior_subset: controls whether to subset prior to centering.
#         w_mean: mean of original embeddings.
#
#     Returns:
#
#     """
#     assert len(walk_features.shape) == 3
#     assert len(original_features.shape) == 2
#     assert type(code_selection) is list or code_selection.__eq__(None)
#     assert mode in ['end', 'diff']
#
#     original_features_p, walk_features_p = original_features, walk_features
#     if centering:
#         walk_features_p = _feature_centering(walk_features_p,
#                                              original_features_p,
#                                              code_selection,
#                                              direction_selection,
#                                              centering,
#                                              prior_subset,
#                                              w_mean)
#
#     if not centering and prior_subset:
#         features = walk_features
#         # For just magnitudes computation.
#         if direction_selection:
#             features = features[:, direction_selection]
#
#         if code_selection:
#             features = features[code_selection]
#             original_features = original_features[code_selection]
#
#     if mode == 'end':
#         features = walk_features_p
#     elif mode == 'diff':
#         if not centering and prior_subset:
#             features = features - original_features.unsqueeze(1).repeat(1, features.shape[1], 1)
#         else:
#             features = walk_features_p - original_features.unsqueeze(1).repeat(1, walk_features_p.shape[1], 1)
#
#     if normalize:
#         features = features / torch.norm(features, dim=-1, keepdim=True)
#
#     return features