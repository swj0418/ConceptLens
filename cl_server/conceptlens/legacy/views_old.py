# # Create your views here.
# import os
# import hashlib
# import json
#
# from scipy.cluster.hierarchy import linkage
#
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
#
# import seaborn as sns
# import numpy as np
#
# from .utilities import postprocess_features
# from .utilities.features import compute_mean_features_pairwise_distances, compute_mean_features, \
#     compute_cosine_distance
# from .utilities.readers import read_walk_features
# from .utilities.readers import _read_experiment_uids
# from .utilities import trees
# from .trees import *
# from .utilities.trees import compute_leaf_weave, append_weave_tree
#
# # served_data_root = '/home/sangwon/Work/concept-lens/data_walking/output'
# SERVED_DATA_ROOT = 'served_data'
#
#
# def weave_score_pairwise(membership, membership_pairs):
#     modified_ones = [1 if a != b else 0 for (a, b) in membership_pairs]
#     modified_sum = sum(modified_ones)
#     weave_score = modified_sum / (math.comb(len(membership), 2) + 1)
#     return weave_score
#
#
# def str2bool(v):
#     if type(v) == bool:
#         return v
#
#     return v.lower() in ("yes", "true", "t", "1", "True")
#
#
# @csrf_exempt
# def get_setting(request):
#     """
#
#     Returns:
#
#     """
#     data = json.loads(request.body)
#     experiment_name = data['experiment_name']
#
#     # Read Data
#     root = os.path.join('served_data', experiment_name)
#
#     # Read config
#     with open(os.path.join(root, 'setting.json'), 'r') as file:
#         setting = json.load(file)
#
#     response = json.dumps(setting, indent=2)
#     return JsonResponse(response, safe=False)
#
#
# """
# For debugging purposes
# """
#
#
# def _compute_mean_feature_difference(raw_features):
#     features = [postprocess_features(feature, None, 'diff', normalize=False) for feature in raw_features]
#     for i, feature in enumerate(features):
#         direction_average = torch.mean(feature, dim=0)
#         set_average_vector = torch.mean(direction_average, dim=0)
#         set_average_diff_norm = torch.norm(set_average_vector)
#         # print(f"\t\t DEBUG: Dataset {i} mean feature difference: {set_average_diff_norm}")
#
#
# def _plot_histogram_nodes(values, title):
#     sns.set(style="darkgrid")
#     sns.histplot(data=values, label=title, bins=10)
#
#
# def _direction_hierarchical_clustering_initialization(experiment_names, dist_metric=None, code_selection_clustering=None):
#     # Determine if pre-computed data exist
#     uids = _read_experiment_uids(experiment_names)
#     uid_combination = ''.join(uids).encode()
#     uid_hash = hashlib.md5(uid_combination).hexdigest()
#
#     precomputed_root = os.path.join(SERVED_DATA_ROOT, 'precomputed')
#     os.makedirs(precomputed_root, exist_ok=True)
#     feature_end_fp = os.path.join(SERVED_DATA_ROOT, 'precomputed', f'end-{uid_hash}')
#     feature_diff_fp = os.path.join(SERVED_DATA_ROOT, 'precomputed', f'diff-{uid_hash}')
#
#     # Feature End Tensor
#     if os.path.exists(feature_end_fp) and not code_selection_clustering:
#         precomputed_data = torch.load(feature_end_fp, map_location='cpu')
#         mean_clu_features = precomputed_data['mean_clu_features']
#         cumulative_dfm_size = precomputed_data['cumulative_dfm_size']
#     else:
#         # Compute and save
#         raw_features = read_walk_features(experiment_names, SERVED_DATA_ROOT)
#
#         # Post-process features [dataset_idx, TENSOR(n_code, n_direction, feature_dim)]
#         clustering_features = [postprocess_features(f, None, 'end') for f in raw_features]
#         cumulative_dfm_size = _compute_cumulative_dfm_size(clustering_features)
#
#         # Process features for clustering => End
#         if code_selection_clustering:
#             clustering_features = [X[code_selection_clustering, :] for X in clustering_features]
#             # _, mean_clu_features = _compute_mean_features(clustering_features)
#             mean_clu_features = compute_mean_features_pairwise_distances(clustering_features, dist_metric=dist_metric)
#         else:
#             # _, mean_clu_features = _compute_mean_features(clustering_features)
#             mean_clu_features = compute_mean_features_pairwise_distances(clustering_features, dist_metric=dist_metric)
#             torch.save({
#                 'mean_clu_features': mean_clu_features,
#                 'cumulative_dfm_size': cumulative_dfm_size
#             }, feature_end_fp)
#
#     # Feature Diff Tensor
#     if os.path.exists(feature_diff_fp) and not code_selection_clustering:
#         precomputed_data = torch.load(feature_diff_fp, map_location='cpu')
#         coh_features, coh_features_var, mean_coh_features = (precomputed_data['coh_features'],
#                                                              precomputed_data['coh_features_var'],
#                                                              precomputed_data['mean_coh_features'])
#     else:
#         # Compute and save
#         raw_features = read_walk_features(experiment_names, SERVED_DATA_ROOT)
#         coherence_features = [postprocess_features(f, None, 'diff') for f in raw_features]
#         coherence_features_var = [postprocess_features(f, None, 'diff') for f in raw_features]
#
#         # Process features for coherence computation => Diff => Not filtered here
#         if code_selection_clustering:
#             coherence_features = [X[code_selection_clustering, :] for X in coherence_features]
#             coh_features, mean_coh_features = compute_mean_features(coherence_features)
#             coh_features_var = compute_cosine_distance(coherence_features_var)
#         else:
#             coh_features, mean_coh_features = compute_mean_features(coherence_features)
#             coh_features_var = compute_cosine_distance(coherence_features_var)
#             # torch.save({
#             #     'coh_features': coh_features,
#             #     'coh_features_var': coh_features_var,
#             #     'mean_coh_features': mean_coh_features
#             # }, feature_diff_fp)
#
#     feature_edit_dist_mean = torch.mean(torch.norm(coh_features, dim=-1), dim=0)  # [n_direction]
#     feature_edit_dist_std = torch.std(torch.norm(coh_features, dim=-1), dim=1)  # [n_direction]
#
#     # Average magnitude and std on all codes & directions.
#     # avg_std, avg_mag = torch.std_mean(feature_edit_dist)
#     avg_mag = torch.mean(feature_edit_dist_mean).item()
#
#
#     # Standard deviation
#     # avg_std = torch.mean(feature_edit_dist_std).item()
#     avg_std = torch.mean(torch.mean(torch.std(coh_features_var))).item()
#
#     return mean_clu_features, mean_coh_features, coh_features, coh_features_var, cumulative_dfm_size, avg_mag, avg_std
#
#
# def coherence_appending(tree, coh_features, coh_features_var, code_selection_coherence):
#     # Compute coherence - from [n_code, n_direction, dim]
#     compute_direction_coherence(tree,
#                                 features=coh_features,
#                                 var_features=coh_features_var,
#                                 filtered_codes=code_selection_coherence)
#
#     # Added for supporting bar plot.
#     leaves = get_tree_children_alt(tree)
#     tree['leaves'] = leaves
#     return None
#
#
# def tree_building(experiment_names,
#                   mean_clu_features,
#                   coh_features,
#                   coh_feature_var,
#                   code_selection_coherence,
#                   cumulative_dfm_size,
#                   clustering_method,
#                   pairwise_metric,
#                   truncated_tree):
#     """
#     If a tree is given as a parameter, it means that only new coherence computation is needed.
#
#     Args:
#         tree:
#
#     Returns:
#
#     """
#     # Clustering
#
#     # z = linkage(mean_clu_features.numpy(), clustering_method, metric=pairwise_metric, optimal_ordering=True)
#     z = linkage(mean_clu_features.numpy(), clustering_method, optimal_ordering=True)
#
#     # Process tree
#     tree = trees.create_tree_general(z, cum_dfm_size=cumulative_dfm_size, experiment_names=experiment_names)
#     tree = count_concat_tree_general(tree, n_distinct=len(cumulative_dfm_size))
#
#     # Compute coherence - from [n_code, n_direction, dim]
#     mgs, sts = compute_direction_coherence(tree,
#                                            features=coh_features,
#                                            var_features=coh_feature_var,
#                                            filtered_codes=code_selection_coherence)
#
#     std_min = min(sts)
#     std_median = torch.median(torch.tensor(sts))
#     std_quan = torch.quantile(torch.tensor(sts), q=.2)
#     # print(std_min, std_median, std_quan)
#
#     # Added for supporting bar plot.
#     leaves = get_tree_children(tree)
#     tree['leaves'] = leaves
#
#     # Truncated tree
#     node_size_min = coh_features.shape[1] / 20  # More reliable.
#     if str2bool(truncated_tree):
#         # create_truncated_tree_by_variance(tree, std_quan, node_size_min)
#         create_foresaw_truncated_tree(tree, node_size_min)
#
#     """
#     TODO:
#
#     magnitudes and stds are not from visual leaf nodes. They are from true leaf nodes. This may need to change if things
#     do not look great in the visualization.
#     """
#     return tree, mgs, sts
#
#
# def _direction_hierarchical_clustering(experiment_names,
#                                        previous_tree,
#                                        code_selection_clustering,
#                                        code_selection_coherence,
#                                        pairwise_metric='cosine',
#                                        clustering_method='ward',
#                                        truncated_tree=True):
#     """
#
#     Args:
#         experiment_names:
#         code_selection_clustering: code subset for clustering.
#         code_selection_coherence:  code subset for computing coherence metrics.
#         pairwise_metric:
#         clustering_method:
#         truncated_tree:
#
#     Returns:
#
#     """
#     torch.set_printoptions(precision=3)
#     # Read data
#     if code_selection_coherence:
#         code_selection_coherence = sorted(code_selection_coherence)
#     if code_selection_clustering:
#         code_selection_clustering = sorted(code_selection_clustering)
#
#     if not previous_tree:  # initial tree building or re-clustering
#         if code_selection_clustering:  # re-clustering
#             # Re-clustering requires *Original* tree and a new tree. So I need to be building two trees
#
#             # Original (initialization) tree
#             mean_clu_featuresO, mean_coh_featuresO, coh_featuresO, coh_feature_var0, cumulative_dfm_sizeO, avg_magO, avg_stdO = \
#                 _direction_hierarchical_clustering_initialization(experiment_names, pairwise_metric)
#
#             original_tree, mags, stds = tree_building(experiment_names,
#                                                       mean_clu_featuresO,
#                                                       coh_featuresO,
#                                                       coh_feature_var0,
#                                                       code_selection_coherence,
#                                                       cumulative_dfm_sizeO,
#                                                       clustering_method,
#                                                       pairwise_metric,
#                                                       truncated_tree)
#
#             # New tree
#             mean_clu_features, mean_coh_features, coh_features, coh_feature_var, cumulative_dfm_size, avg_mag, avg_std = \
#                 _direction_hierarchical_clustering_initialization(experiment_names, pairwise_metric,
#                                                                   code_selection_clustering)
#
#             new_tree, mags, stds = tree_building(experiment_names,
#                                                  mean_clu_features,
#                                                  coh_features,
#                                                  coh_feature_var,
#                                                  code_selection_coherence,
#                                                  cumulative_dfm_size,
#                                                  clustering_method,
#                                                  pairwise_metric,
#                                                  truncated_tree)
#
#             weave_scores = compute_leaf_weave(original_tree, new_tree)  # Right direction
#             append_weave_tree(new_tree, weave_scores)                   # Right direction
#             tree = new_tree
#         else:
#             (mean_clu_features,
#              mean_coh_features,
#              coh_features,
#              coh_feature_var,
#              cumulative_dfm_size,
#              avg_mag,
#              avg_std) = _direction_hierarchical_clustering_initialization(experiment_names, pairwise_metric)
#
#             new_tree, mags, stds = tree_building(experiment_names,
#                                                  mean_clu_features,
#                                                  coh_features,
#                                                  coh_feature_var,
#                                                  code_selection_coherence,
#                                                  cumulative_dfm_size,
#                                                  clustering_method,
#                                                  pairwise_metric,
#                                                  truncated_tree)
#
#             tree = new_tree
#
#     else:  # Brushing&Clicking
#         # Only coherence needs to be computed, raw features are needed. Nothing can be precomputed
#         raw_features = read_walk_features(experiment_names)
#         coherence_features = [postprocess_features(f, code_selection_coherence, 'diff') for f in raw_features]
#         coherence_features_var = [postprocess_features(f, code_selection_coherence, 'diff') for f in raw_features]
#         coh_features, mean_coh_features = compute_mean_features(coherence_features)  # Coherence feat of magnitude.
#         coh_features_var = compute_cosine_distance(coherence_features_var)  # Coherence feat of variance.
#
#         coherence_appending(previous_tree, coh_features, coh_features_var, code_selection_coherence)
#
#         # feature_edit_dist_mean = torch.mean(torch.norm(coh_features, dim=-1), dim=0)  # [n_direction]
#         # feature_edit_dist_std = torch.std(torch.norm(coh_features, dim=-1), dim=0)  # [n_direction]
#         #
#         # # Average magnitude and std on all codes & directions.
#         # # avg_std, avg_mag = torch.std_mean(feature_edit_dist)
#         # avg_mag = torch.mean(feature_edit_dist_mean).item()
#         # avg_std = torch.mean(feature_edit_dist_std).item()
#
#         tree = previous_tree  # Server-side sends correctly updated tree. 20240103
#
#         # Re-append leaves
#         # reappend_leaves(tree)
#
#         # TODO: Is this okay...? I am not using them anyways while brushing and clicking.
#         avg_mag = 0
#         avg_std = 0
#         mags = [0, 0]
#         stds = [0, 0]
#
#     return tree, avg_mag, avg_std, mags, stds
#
#
# def _code_hierarchical_clustering(experiment_names, directions, clustering_method, truncated_tree):
#     torch.set_printoptions(precision=3)
#
#     clustering_method = 'ward'
#
#     # Get raw features
#     raw_features = read_walk_features(experiment_names)
#
#     # Post-process features [dataset_idx, TENSOR(n_code, n_direction, feature_dim)]
#     clustering_features = [postprocess_features(f, None, 'start') for f in raw_features]
#     coherence_features =  [postprocess_features(f, None, 'diff') for f in raw_features]
#
#     # Process features for clustering => End
#     clu_features, _ = compute_mean_features(clustering_features)
#     clu_features = clu_features[:, 0, :]
#     # clu_features = torch.mean(clu_features, dim=1)  # This creates a small perturbation that leads to diff code clu.
#
#     # Process features for coherence computation => Diff => Not filtered here
#     coh_features, mean_coh_features = compute_mean_features(coherence_features)
#     coh_features_var = compute_cosine_distance(coherence_features)
#
#     # Cluster
#     z = linkage(clu_features.numpy(), clustering_method, optimal_ordering=True)
#     tree = create_tree(z, 1, 1)
#     tree = count_tree(tree)
#
#     # ========================================== Coherence computation ==============================================
#     if len(directions) == 0:  # Conditional computation
#         directions = None
#
#     # This is distance
#     edit_distance = torch.norm(coh_features, dim=-1)  # [n_code, n_direction]
#     standard_deviations, distance_magnitudes = compute_code_coherence(tree,
#                                                                       edit_distance,
#                                                                       coh_features_var,
#                                                                       directions)
#
#     # Added for supporting bar plot.
#     leaves = get_tree_children(tree)
#     tree['leaves'] = leaves
#
#     # Truncated tree
#     node_size_min = clu_features.shape[0] / 20
#     if str2bool(truncated_tree):
#         create_foresaw_truncated_tree(tree, node_size_min)
#
#     # Compute averages / minmax
#     # feature_edit_dist = torch.norm(torch.mean(coh_features, dim=0), dim=-1)  # [n_direction]
#     feature_edit_dist_mean = torch.mean(torch.norm(coh_features, dim=-1), dim=0)  # [n_direction]
#     feature_edit_dist_std = torch.std(torch.norm(coh_features, dim=-1), dim=1)  # [n_direction]
#
#     # Average magnitude and std on all codes & directions.
#     # avg_std, avg_mag = torch.std_mean(feature_edit_dist)
#     avg_mag = torch.mean(feature_edit_dist_mean).item()
#     avg_std = torch.mean(feature_edit_dist_std).item()
#
#     return tree, avg_mag, avg_std
#
#
# def _during_direction_selection(diff_features, directions, selected_codes, dist_metric='cosine'):
#     # Process diff vector => [code, selected directions, feature_dim]
#     features = torch.cat(diff_features, dim=1)[:, directions]
#
#     # Re-center
#     centering_mean = torch.mean(torch.mean(features, dim=0), dim=0)
#     features = features - centering_mean
#
#     ## === Clustering ===
#     # # [c, d, f] => [d, c, f]
#     # features = torch.swapaxes(features, 0, 1)
#     #
#     # # Compute distance of between codes across all directions.
#     # pds = []
#     # for feat in features:
#     #     d = scipy.spatial.distance.pdist(feat, metric=dist_metric)  # I need a condensed form
#     #     pds.append(torch.tensor(d))
#     #
#     # pds = torch.stack(pds)
#     # average_pd = torch.mean(pds, dim=0)  # Averaged over directions
#
#     ## === Projection ===
#     mean_features = torch.mean(features, dim=1)  # Mean over directions
#
#     projector = umap.UMAP()
#     projected_features = projector.fit_transform(mean_features.numpy())
#
#     # hacked labels
#     labels = np.array([0 if k not in selected_codes else 1 for k in range(200)])
#     # print(labels)
#
#     # umap.plot.points(projector, labels=labels, theme='fire')  # edge_bundling='hammer'
#
#     # sns.scatterplot(data=projected_features)
#     # plt.savefig('projections.png')
#     return projected_features
#
#
# def _code_just_coherence(experiment_names, previous_tree, directions, selected_codes):
#     # Get raw features
#     raw_features = read_walk_features(experiment_names)
#
#     # Post-process features [dataset_idx, TENSOR(n_code, n_direction, feature_dim)]
#     coherence_features = [postprocess_features(f, None, 'diff') for f in raw_features]
#     # Process features for coherence computation => Diff => Not filtered here
#     coh_features, mean_coh_features = compute_mean_features(coherence_features)
#     coh_features_var = compute_cosine_distance(coherence_features)
#
#     # Debug: During code-selection, cluster codes based on the difference vector and show them in a scatter plot
#     # _during_direction_selection(coherence_features, directions, selected_codes)
#
#     # ========================================== Coherence computation ==============================================
#     if len(directions) == 0:  # Conditional computation
#         directions = None
#
#     # This is distance
#     edit_distance = torch.norm(coh_features, dim=-1)  # [n_code, n_direction]
#     compute_code_coherence(previous_tree,
#                            edit_distance,
#                            coh_features_var,
#                            directions)
#
#     # Compute averages / minmax
#     # feature_edit_dist = torch.norm(torch.mean(coh_features, dim=0), dim=-1)  # [n_direction]
#     feature_edit_dist_mean = torch.mean(torch.norm(coh_features, dim=-1), dim=0)  # [n_direction]
#     feature_edit_dist_std = torch.std(torch.norm(coh_features, dim=-1), dim=0)  # [n_direction]
#
#     # Average magnitude and std on all codes & directions.
#     # avg_std, avg_mag = torch.std_mean(feature_edit_dist)
#     avg_mag = torch.mean(feature_edit_dist_mean).item()
#     avg_std = torch.mean(feature_edit_dist_std).item()
#
#     tree = previous_tree
#     return tree, avg_mag, avg_std
#
#
# @csrf_exempt
# def direction_hierarchical_clustering(request):
#     """
#
#     Args:
#         request:
#
#     Returns:
#
#     """
#     torch.set_printoptions(precision=3, sci_mode=False)
#     data = json.loads(request.body)
#
#     experiment_names = data['experiment_names']
#     previous_tree = data['tree']
#     code_selection_clustering = data['code_selection_clustering']
#     code_selection_coherence = data['code_selection_coherence']
#     pairwise_metric = data['pairwise_metric']
#     clustering_method = data['clustering_method']
#     truncated_tree = data['truncated_tree']
#
#     tree, avg_magnitude, avg_std, mags, stds = _direction_hierarchical_clustering(experiment_names,
#                                                                                   previous_tree,
#                                                                                   code_selection_clustering,
#                                                                                   code_selection_coherence,
#                                                                                   pairwise_metric,
#                                                                                   clustering_method,
#                                                                                   truncated_tree)
#
#     mags_std = np.std(mags)
#     stds_std = np.std(stds)
#
#     resp = json.dumps({
#         'tree': tree,
#         'timestamp': time.time(),
#         'avgMagnitude': float(avg_magnitude),
#         'avgStd': float(avg_std),
#         'magsStd': mags_std,
#         'stdsStd': stds_std
#     }, indent=2)
#
#     return JsonResponse(resp, safe=False)
#
#
# @csrf_exempt
# def code_hierarchical_clustering(request):
#     """
#
#     Returns:
#
#     """
#     torch.set_printoptions(precision=3, sci_mode=False)
#
#     data = json.loads(request.body)
#     experiment_names = data['experiment_names']
#     previous_tree = data['tree']
#     directions = data['directions']
#     # metric = data['metric']
#     clustering_method = data['clustering_method']
#     truncated_tree = data['truncated_tree']
#
#     selected_codes = []
#     try:
#         selected_codes = data['codes']
#         selected_codes = [c['name'] for c in selected_codes]
#         # print("Selected Codes: ", selected_codes)
#     except:
#         pass
#
#     if not previous_tree:
#         tree, avg_mag, avg_std = _code_hierarchical_clustering(experiment_names,
#                                                                directions,
#                                                                clustering_method,
#                                                                truncated_tree)
#     else:
#         tree, avg_mag, avg_std = _code_just_coherence(experiment_names, previous_tree, directions, selected_codes)
#
#         # Added for supporting bar plot. # There is a problem with "leaf" variable when tree is rebuilt.
#         leaves = get_tree_children_alt(tree)
#         tree['leaves'] = leaves
#
#     resp = json.dumps({
#         'tree': tree,
#         'timestamp': time.time(),
#         'avgMagnitude': float(avg_mag),
#         'avgStd': float(avg_std)
#     }, indent=2)
#
#     return JsonResponse(resp, safe=False)
#
#
# # dfs to gather all leaf nodes
# def gather_truncated_tree_leaves(node):
#     if node['leaf']:  # This fails to work when tree is truncated.
#         if len(node['leaves']) > 0:  # Vis leaves for brushing & clicking.
#             return [n['name'] for n in node['leaves']]
#         else:
#             return [node['name']]
#     else:
#         leafs = []
#         for child in node['children']:
#             leafs.extend(gather_truncated_tree_leaves(child))
#         return leafs
#
#
# def compute_direction_coherence(tree, features, var_features, filtered_codes):
#     """
#     Computes std of the intermediate nodes, until the stop depth.
#
#     This is an in place operation.
#
#     Args:
#         tree:
#         stop_depth:
#
#     Returns:
#
#     """
#     # all_features = features.clone()
#     if filtered_codes and len(filtered_codes) < features.shape[0]:  # [n_code, n_direction, n_dim]
#         features = features[filtered_codes]
#         var_features = var_features[filtered_codes]
#     else:
#         pass  # It is already filtered.
#
#     # print(f"Computing direction coherence on tensor shape {features.shape}")
#     a_tree_magnitudes, tree_magnitudes = [], []
#     a_tree_stds, tree_stds = [], []
#
#     # Post-order traverse
#     def post_order_traverse(node):
#         children = node['children']
#         if len(children) > 0:
#             for child in children:
#                 post_order_traverse(child)
#
#         # Gather all leaf nodes - These are direction names - inefficient. Traversing too many times.
#         leafs = sorted(gather_truncated_tree_leaves(node))
#
#         subset = features[:, leafs]  # [n_code, (sub_direction), n_dim]
#         varsubset = var_features[:, leafs]  # [n_code, (sub_direction), n_dim]
#
#         # Code-wise averaging
#         # edit_distances_std = torch.std(torch.mean(torch.norm(subset, dim=-1), dim=0), dim=0).item()  # [n_code, (dir)] => [(dir)]
#         # edit_distances_mean = torch.mean(torch.mean(torch.norm(subset, dim=-1), dim=0), dim=0).item()
#         # std_mean = edit_distances_std
#         # magnitude_mean = edit_distances_mean
#         # if math.isnan(std_mean):
#         #     std_mean = 0
#
#         # if node['leaf']:
#         #     print(torch.norm(subset, dim=-1))
#
#         # Working Method - Std
#         # edit_distances_std = torch.std(torch.norm(subset, dim=-1), dim=1)  # [n_code, (dir)] => [(dir)]
#         # std_mean = torch.mean(edit_distances_std).item()
#
#         # Testing Method - Std, cos
#         edit_distances_std = torch.std(varsubset, dim=(1, 2))
#         std_mean = torch.mean(edit_distances_std).item()
#         # print(" ", std_mean)
#
#         # Working Method - Mean
#         edit_distances_mean = torch.mean(torch.norm(subset, dim=-1), dim=0)
#         magnitude_mean = torch.mean(edit_distances_mean).item()
#
#         # print(node['depth'] * ' ', magnitude_mean, "  ", subset.shape, " ", node['name'])
#         if math.isnan(std_mean):
#             std_mean = 0
#
#         tree_magnitudes.append(magnitude_mean)
#         tree_stds.append(std_mean)
#
#         # Insert data to tree
#         node['magnitude'] = magnitude_mean
#         node['var'] = std_mean
#
#     post_order_traverse(tree)
#     return tree_magnitudes, tree_stds
#
#
# def compute_code_coherence(tree, edit_distance, var_features, filtered_directions=None, stop_depth=16):
#     """
#     Computes std and mean of the edit distance of all tree nodes until stop_depth.
#
#     Args:
#         tree:
#         edit_distance: Tensor shaped by [n_code, n_direction]
#         filtered_directions:
#         stop_depth:
#
#     Returns:
#
#     """
#     standard_deviations = []
#     distance_magnitudes = []
#
#     if filtered_directions:
#         edit_distance = edit_distance[:, filtered_directions]
#         var_features = var_features[:, filtered_directions]
#
#     # Post-order traverse
#     def post_order_traverse(node):
#         # if node['depth'] > stop_depth:
#         #     pass
#         # elif node['depth'] <= stop_depth and not node['leaf']:
#         #     # Continue traversing
#         #     for child in node['children']:
#         #         post_order_traverse(child)
#         children = node['children']
#         if len(children) > 0:
#             for child in children:
#                 post_order_traverse(child)
#
#         # Gather all leaf nodes
#         leafs = gather_truncated_tree_leaves(node)
#         subset = edit_distance[leafs]  # Code subsets => [n_code, n_direction]
#         varsubset = var_features[leafs]
#
#         subset_magnitudes = torch.mean(subset, dim=1)
#         # subset_stds = torch.std(subset, dim=1)
#         subset_stds = torch.std(varsubset, dim=(1, 2))
#
#         # Compute mean and std
#         magnitude_mean = torch.mean(subset_magnitudes).item()
#         std_mean = torch.mean(subset_stds).item()
#         if math.isnan(std_mean):
#             std_mean = 0
#
#         # Insert data to tree
#         node['var'] = std_mean
#         node['magnitude'] = magnitude_mean
#
#     post_order_traverse(tree)
#
#     return standard_deviations, distance_magnitudes
