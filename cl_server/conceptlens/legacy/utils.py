# def compute_direction_coherence(tree, features, filtered_codes, stop_depth, use_dist=False):
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
#     standard_deviations = []
#     distance_magnitudes = []
#
#     if filtered_codes:  # [n_code, n_direction, n_dim]
#         features = features[filtered_codes]
#
#     # Torch tensor
#     # print(torch.backends.mps.is_available())
#     if type(features) == np.ndarray:
#         features = torch.from_numpy(features)
#
#     if torch.cuda.is_available():
#         features.to('cuda')
#
#     # dfs to gather all leaf nodes
#     def gather_leaf(node):
#         if node['leaf']:
#             return [node['name']]
#         else:
#             leafs = []
#             for child in node['children']:
#                 leafs.extend(gather_leaf(child))
#             return leafs
#
#     # Post-order traverse
#     def post_order_traverse(node):
#         if node['depth'] > stop_depth:
#             if not node['leaf']:
#                 for child in node['children']:
#                     post_order_traverse(child)
#         elif node['depth'] <= stop_depth and not node['leaf']:
#             # Post-order Continue traversing
#             for child in node['children']:
#                 post_order_traverse(child)
#
#             # Gather all leaf nodes
#             leafs = sorted(gather_leaf(node))
#
#             # Index the pairwise distance tensor
#             if use_dist:
#                 subset = features[:, leafs]
#             else:
#                 subset = features[:, leafs]
#                 subset = subset[:, :, leafs]
#
#             # Compute std [code, direction] (in case of edit distance)
#             subset = torch.flatten(subset, start_dim=1)
#
#             # [std of direction group]
#             stds = torch.std(subset, dim=1)
#             # print(stds.shape)
#             std = float(torch.mean(stds))
#
#             # Compute distance magnitude
#             magnitude = abs(float(torch.mean(subset)))
#
#             standard_deviations.append(std)
#             distance_magnitudes.append(magnitude)
#
#             # Insert data to tree
#             node['var'] = std
#             node['magnitude'] = magnitude
#
#     post_order_traverse(tree)
#
#     return standard_deviations, distance_magnitudes
#
#
# def compute_code_coherence(tree, pairwise_distance, filtered_directions, stop_depth, use_dist=False):
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
#     standard_deviations = []
#     distance_magnitudes = []
#
#     if filtered_directions:
#         if use_dist:
#             pairwise_distance = pairwise_distance[:, filtered_directions]
#         else:
#             pairwise_distance = pairwise_distance[:, filtered_directions]
#             pairwise_distance = pairwise_distance[:, :, filtered_directions]
#
#     if type(pairwise_distance) == np.ndarray:
#         pairwise_distance = torch.from_numpy(pairwise_distance)
#
#     if torch.cuda.is_available():
#         pairwise_distance.to('cuda')
#
#     # dfs to gather all leaf nodes
#     def gather_leaf(node):
#         if node['leaf']:
#             return [node['name']]
#         else:
#             leafs = []
#             for child in node['children']:
#                 leafs.extend(gather_leaf(child))
#             return leafs
#
#     # Post-order traverse
#     def post_order_traverse(node):
#         if node['depth'] > stop_depth:
#             pass
#         elif node['depth'] <= stop_depth and not node['leaf']:
#             # Continue traversing
#             for child in node['children']:
#                 post_order_traverse(child)
#
#             # Gather all leaf nodes
#             leafs = gather_leaf(node)
#
#             # Index the pairwise distance tensor
#             subset = pairwise_distance[leafs]
#
#             # Compute std
#             stds = torch.tensor([torch.std(c) for c in subset])
#             std = float(torch.mean(stds))
#
#             # Compute distance magnitude
#             magnitude = float(torch.mean(subset))
#
#             standard_deviations.append(std)
#             distance_magnitudes.append(magnitude)
#
#             # Insert data to tree
#             node['var'] = std
#             node['magnitude'] = magnitude
#
#     post_order_traverse(tree)
#
#     return standard_deviations, distance_magnitudes
