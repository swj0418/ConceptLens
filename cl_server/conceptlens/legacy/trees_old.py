# import math
# import time
#
#
# # General Tree Functions
# def get_tree_children(tree):
#     if tree['leaf']:  # Termination condition
#         return [tree]
#
#     container = []
#     for child in tree['children']:
#         result = get_tree_children(child)
#         container.extend(result)
#     return container
#
#
# # General Tree Functions
# def get_tree_children_alt(tree):
#     if len(tree['children']) == 0:  # Termination condition
#         return [tree]
#
#     container = []
#     for child in tree['children']:
#         result = get_tree_children_alt(child)
#         container.extend(result)
#     return container
#
#
# def get_tree_size(tree, names):
#     if tree['leaf']:
#         names.append(tree)
#         return tree['quantity']
#     else:
#         quantity = 0
#         for t in tree['children']:
#             q = get_tree_size(t, names)
#             quantity += q
#         return quantity
#
#
# def create_tree(linked, dfmSize, methodSize):
#     """
#     https://stackoverflow.com/questions/65462806/how-to-parse-data-from-scikitlearn-agglomerative-clustering
#
#     Args:
#         linked:
#
#     Returns:
#
#     """
#     # Count func to recursively find leaf nodes.
#
#     # inner func to recursively walk the linkage matrix
#     def recurTree(tree):
#         k = tree['name']
#         # no children for this node
#         if k not in inter:
#             return
#         for n in inter[k]:
#             # build child nodes
#             leaf = None
#             if n not in inter:
#                 leaf = True
#                 node = {
#                     "name": int(n),
#                     "leaf": leaf,
#                     "parent": int(k),
#                     "treeID": math.floor(n / dfmSize),
#                     "flatIdx": int(n) - math.floor(n / dfmSize) * dfmSize,
#                     "children": [],
#                     "leaves": [],
#                     "trip": 0,
#                     "var": 0,
#                     "magnitude": 0
#                 }
#             else:
#                 leaf = False
#                 node = {
#                     "name": int(n),
#                     "leaf": leaf,
#                     "parent": int(k),
#                     "children": [],
#                     "trip": 0,
#                     "var": 0,
#                     "magnitude": 0
#                 }
#
#             # add to children
#             tree['children'].append(node)
#
#             # next recursion
#             recurTree(node)
#
#     num_rows, num_cols = linked.shape
#     inter = {}
#     i = 0
#
#     # Intermediate nodes
#     for row in linked:
#         i += 1
#         inter[float(i + num_rows)] = [row[0], row[1]]
#
#     tree = {
#         "name": int(i + num_rows),
#         "timestamp": time.time(),
#         "origin": int(i),
#         "leaf": False,
#         "parent": None,
#         "children": [],
#         "leaves": [],
#         "trip": 0,
#         "var": 0,
#         "magnitude": 0
#     }
#
#     # Start recursion
#     recurTree(tree)
#     return tree
#
#
# def count_tree(parent_tree):
#     def depth(node):
#         return max(depth(node['children'][0]) if node['children'][0] else 0, depth(node['children'][1]) if node['children'][1] else 0) + 1
#
#     def post_order_trav(tree, count, depth):
#         leaf = tree["leaf"]
#         if not leaf:
#             tree["depth"] = depth
#             depth += 1
#
#             new_count = 0
#             sub_count_0 = post_order_trav(tree['children'][0], new_count, depth)
#             sub_count_1 = post_order_trav(tree['children'][1], new_count, depth)
#             tree["quantity"] = sub_count_0 + sub_count_1
#
#             return sub_count_0 + sub_count_1
#         else:
#             tree["depth"] = depth
#             tree["quantity"] = 1
#
#             count += 1
#             return count
#
#     post_order_trav(parent_tree, 0, 0)
#
#     return parent_tree
#
#
# def count_tree_target(parent_tree, treeID):
#     def depth(node):
#         return max(depth(node['children'][0]) if node['children'][0] else 0, depth(node['children'][1]) if node['children'][1] else 0) + 1
#
#     def post_order_trav(tree, count, tc, depth):
#         leaf = tree["leaf"]
#         if not leaf:
#             tree["depth"] = depth
#             depth += 1
#
#             sub_count_0, tc0 = post_order_trav(tree['children'][0], 0, 0, depth)
#             sub_count_1, tc1 = post_order_trav(tree['children'][1], 0, 0, depth)
#             tree["quantity"] = sub_count_0 + sub_count_1
#             tree["tquantity"] = tc0 + tc1
#
#             return sub_count_0 + sub_count_1, tc0 + tc1
#         else:
#             tree["depth"] = depth
#             if int(tree['treeID']) == int(treeID):
#                 tree["quantity"] = 1
#                 count += 1
#             else:
#                 tree["quantity"] = 0
#             tc += 1
#             return count, tc
#
#     post_order_trav(parent_tree, 0, 0, 0)
#
#     return parent_tree
#
#
# def count_concat_tree(parent_tree):
#     def depth(node):
#         return max(depth(node['children'][0]) if node['children'][0] else 0, depth(node['children'][1]) if node['children'][1] else 0) + 1
#
#     def post_order_trav(tree, count, count_a, count_b, depth):
#         leaf = tree["leaf"]
#         if not leaf:
#             tree["depth"] = depth
#             depth += 1
#
#             new_count_a, new_count_b = 0, 0
#             counta, sub_count_0a, sub_count_0b = post_order_trav(tree['children'][0], 0, new_count_a, new_count_b, depth)
#             countb, sub_count_1a, sub_count_1b = post_order_trav(tree['children'][1], 0, new_count_a, new_count_b, depth)
#             tree["quantity"] = counta + countb
#             tree["quantity_a"] = sub_count_0a + sub_count_1a
#             tree["quantity_b"] = sub_count_0b + sub_count_1b
#
#             return counta + countb, sub_count_0a + sub_count_1a, sub_count_0b + sub_count_1b
#         else:
#             tree["depth"] = depth
#             if tree['treeID'] == 0:
#                 tree["quantity_a"] = 1
#                 tree["quantity"] = 1
#                 count_a += 1
#             else:
#                 tree["quantity_b"] = 1
#                 tree["quantity"] = 1
#                 count_b += 1
#             count += 1
#
#             return count, count_a, count_b
#
#     post_order_trav(parent_tree, 0, 0, 0, 0)
#
#     return parent_tree
#
# #
# # def create_tree_general(linked, cum_dfm_size, experiment_names=None):
# #     """
# #     https://stackoverflow.com/questions/65462806/how-to-parse-data-from-scikitlearn-agglomerative-clustering
# #
# #     Args:
# #         linked:
# #
# #     Returns:
# #
# #     """
# #     cum_dfm_size = torch.cat([torch.tensor([0]), cum_dfm_size])
# #     # sequential search
# #
# #     # inner func to recursively walk the linkage matrix
# #     def recurTree(tree):
# #         k = tree['name']
# #         # no children for this node
# #         if k not in inter:
# #             return
# #         for n in inter[k]:
# #             # build child nodes
# #             if n not in inter:  # Leaf
# #                 name = int(n)
# #                 # Determine treeID. I have cumulative summation e.g., [180, 230, 380] for dfms of size [180, 50, 150].
# #                 treeID = search_id(name, cum_dfm_size)
# #                 flatIdx = name
# #                 if treeID > 0:
# #                     flatIdx = int(name - cum_dfm_size[treeID])
# #
# #                 node = {
# #                     "name": name,
# #                     "leaf": True,
# #                     "parent": int(k),
# #                     "treeID": treeID,
# #                     "expName": experiment_names[treeID],
# #                     "flatIdx": flatIdx,
# #                     "children": [],
# #                     "leaves": [],
# #                     "trip": 0,
# #                     "var": 0,
# #                     "magnitude": 0
# #                 }
# #             else:  # Non-leaf
# #                 node = {
# #                     "name": int(n),
# #                     "leaf": False,
# #                     "parent": int(k),
# #                     "children": [],
# #                     "leaves": [],
# #                     "var": 0,
# #                     "magnitude": 0,
# #                     "trip": 0
# #                 }
# #
# #             # add to children
# #             tree['children'].append(node)
# #
# #             # next recursion
# #             recurTree(node)
# #
# #     # Intermediate nodes
# #     for row in linked:
# #         i += 1
# #         inter[float(i + num_rows)] = [row[0], row[1]]
# #
# #     tree = {
# #         "name": int(i + num_rows),
# #         "timestamp": time.time(),
# #         "origin": int(i),
# #         "leaf": False,
# #         "parent": None,
# #         "children": [],
# #         "leaves": [],
# #         "trip": 0,
# #         "var": 0,
# #         "magnitude": 0
# #     }
# #
# #     # Start recursion
# #     recurTree(tree)
# #     return tree
#
#
# # def count_concat_tree_general(parent_tree, n_distinct):
# #     def depth(node):
# #         return max(depth(node['children'][0]) if node['children'][0] else 0, depth(node['children'][1]) if node['children'][1] else 0) + 1
# #
# #     def post_order_trav(tree, total_count, distinct_count, depth):
# #         leaf = tree["leaf"]
# #         if not leaf:
# #             tree["depth"] = depth
# #             depth += 1
# #
# #             new_total_count, new_distinct_count = 0, [0 for _ in range(n_distinct)]
# #             count_a, sub_distinct_count_a = post_order_trav(tree['children'][0], new_total_count, new_distinct_count, depth)
# #             count_b, sub_distinct_count_b = post_order_trav(tree['children'][1], new_total_count, new_distinct_count, depth)
# #             tree["quantity"] = count_a + count_b
# #
# #             counter = 0
# #             for a, b in zip(sub_distinct_count_a, sub_distinct_count_b):
# #                 tree[f"quantity_{counter}"] = a + b
# #
# #             return count_a + count_b, sub_distinct_count_a + sub_distinct_count_b
# #         else:
# #             tree["depth"] = depth
# #             tree[f"quantity_{tree['treeID']}"] = 1
# #             tree["quantity"] = 1
# #
# #             distinct_count[tree['treeID']] += 1
# #             total_count += 1
# #
# #             return total_count, distinct_count
# #
# #     total_count, distinct_count = 0, [0 for _ in range(n_distinct)]
# #     post_order_trav(parent_tree, total_count, distinct_count, depth=0)
# #
# #     return parent_tree
#
#
# def create_truncated_tree(tree, treemin, treemax):
#     true_leaf_nodes = []
#     quantity = get_tree_size(tree, true_leaf_nodes)
#
#     if treemin < quantity < treemax:
#         tree['true_leaf_nodes'] = true_leaf_nodes
#         tree['leaf'] = True
#         # tree['children'] = None
#         return
#     elif treemax <= quantity:
#         for children in tree['children']:
#             create_truncated_tree(children, treemin, treemax)
#     elif quantity <= treemin:
#         # Do nothing.
#         pass
#         # (Ideally) raise Exception("This should not be reachable")
#
#
# # def create_foresaw_truncated_tree_old(tree, treemin, treemax):
# #     current_level_true_leaf_nodes = []
# #     current_level_quantity = get_tree_size(tree, current_level_true_leaf_nodes)\
# #
# #     for child in tree['children']:
# #         true_leaf_nodes = []
# #         quantity = get_tree_size(child, true_leaf_nodes)
# #         if treemin < quantity < treemax:
# #             # child['true_leaf_nodes'] = true_leaf_nodes
# #             # child['visleaf'] = True
# #             # child['leaf'] = True
# #             create_foresaw_truncated_tree(child, treemin, treemax)
# #         elif treemax <= quantity:
# #             create_foresaw_truncated_tree(child, treemin, treemax)
# #         elif quantity <= treemin:
# #             # Put a stop here
# #             # tree['true_leaf_nodes'] = current_level_true_leaf_nodes
# #             # tree['visleaf'] = True
# #
# #             child['leaf'] = True
#
# def create_truncated_tree_by_variance(tree, variance_threshold, node_size_min):
#     """
#     Split continuation criteria is based on the variance of the split node. If, after splitting, node variance
#     (inconsistency -> std) goes under certain level, do not split anymore.
#
#     Args:
#         tree:
#
#     Returns:
#
#     """
#     child_nodes = tree['children']
#
#     for child in child_nodes:
#         quantity, depth = child['quantity'], child['depth']
#         node_variance = child['var']
#         # print(" " * depth, node_variance)
#         if child['leaf']:
#             child['leaves'] = []
#             child['visleaves'] = []
#             child['leaf'] = True
#         else:
#             if quantity < node_size_min:
#                 # Get child leaf nodes
#                 leaves = get_tree_children(child)
#
#                 # Set list and prune
#                 child['leaves'] = leaves
#                 child['visleaves'] = []
#
#                 # Set boolean checks
#                 child['leaf'] = True
#             else:
#                 if variance_threshold <= node_variance:
#                     create_truncated_tree_by_variance(child, variance_threshold, node_size_min)
#                 elif node_variance < variance_threshold:
#                     # Get child leaf nodes
#                     leaves = get_tree_children(child)
#
#                     # Set list and prune
#                     child['leaves'] = leaves
#                     child['visleaves'] = []
#
#                     # Set boolean checks
#                     child['leaf'] = True
#
#
# def create_foresaw_truncated_tree(tree, treemin):
#     """
#     If "leaf" key in the tree structure is set to be true, it messes up things in the visualization code w.r.t. tree
#     visualization (hierarchyScale). It is not ideal to change the hierarchyScale since it expects a raw tree, which
#     is a more general case.
#
#     For tree children aggregation operation, we want the tree to maintain its raw structure. I can have a workaround,
#     but I ran into issue before doing that.
#
#     Args:
#         tree:
#         treemin:
#
#     Returns:
#
#     """
#     child_nodes = tree['children']
#
#     for child in child_nodes:
#         quantity, depth = child['quantity'], child['depth']
#         # print(" " * depth, quantity)
#         if child['leaf']:
#             child['leaves'] = []
#             child['visleaves'] = []
#             child['leaf'] = True
#         else:
#             if treemin <= quantity:
#                 # Added for supporting bar plot.
#                 leaves = get_tree_children(child)
#                 child['leaves'] = leaves
#
#                 create_foresaw_truncated_tree(child, treemin)
#             elif quantity < treemin:
#                 # Get child leaf nodes
#                 leaves = get_tree_children(child)
#
#                 # Set list and prune
#                 child['leaves'] = leaves
#                 child['visleaves'] = []
#
#                 # Set boolean checks
#                 child['leaf'] = True
#
#
# def reappend_leaves(tree):
#     child_nodes = tree['children']
#
#     for child in child_nodes:
#         if child['leaf']:
#             pass
#         else:
#             leaves = get_tree_children_alt(child)
#             print(leaves)
#
#             # Set list and prune
#             child['leaves'] = leaves
#
#
