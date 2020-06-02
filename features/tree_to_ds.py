from features.tree_to_fp import linearize_tree_artifical

import numpy as np



def assign_sc_to_tree(sc_dict, tree):
    sc = sc_dict[tree['smiles']]
    tree['sc'] = sc
    if tree['child']:
        tree['child'] = [assign_sc_to_tree(sc_dict, c) for c in tree['child']]
    return tree

def convert_tree_to_sc(tree, max_step=10, reaction_sc=False):
    '''
    :param tree:
    :param max_step:
    :param reaction_sc: Boolean, if True, return reaction sc as (p_sc - max(r_sc)), otherwise, return compound sc
    :return:
    '''
    linear_trees, num_linear_trees = linearize_tree_artifical(tree, return_sc=True)

    tree_compound_scs = np.zeros((num_linear_trees, max_step+1))    # N_compounds = N_reactions + 1
    for k, lt in enumerate(linear_trees):
        for i in range(len(lt)):
            all_compounds = [lt[i]['main']] + lt[i]['other']
            all_sc = [x for x in all_compounds if x]

            tree_compound_scs[k, i] = max(all_sc)
    return tree_compound_scs, num_linear_trees


def convert_one_record_to_sc(record, max_step=10, reaction_sc=False):
    '''

    :param sc_dict:
    :param record:
    :param max_step:
    :param reaction_sc:
    :return:
    '''

    compound_scs_batch = np.zeros((record['num_linear_trees'], max_step+1))
    num_linear_trees_batch = []

    pathways = [record['true_data']]
    pathways.extend(record['generated_paths'])

    next_idx = 0
    for i, pathway in enumerate(pathways):
        tree_compound_scs, num_linear_trees = convert_tree_to_sc(
                                                                 pathway['tree'],
                                                                 max_step)
        compound_scs_batch[next_idx:next_idx + num_linear_trees, :] = tree_compound_scs
        num_linear_trees_batch.append(num_linear_trees)

        next_idx += num_linear_trees

    return compound_scs_batch, np.array(num_linear_trees_batch), len(pathways)


def merge_into_batch(batch_inputs):
    compound_scs_batch, num_linear_trees, num_trees = zip(*batch_inputs)

    compound_scs_batch = np.concatenate(compound_scs_batch, axis=0)
    num_linear_trees = np.concatenate(num_linear_trees)

    compound_scs_batch = compound_scs_batch.astype('float32')

    return compound_scs_batch, num_linear_trees.tolist(), num_trees


def find_all_trees_ds(tree):
    if len(tree['child']) == 0:
        return [[tree]], 1, [tree['sc']], []
    return_tree = []
    num_nodes = 1
    bottom_scs = []
    node_scs = [tree['sc']]
    for c in tree['child']:
        paths, num_sub_nodes, bottom_sc, node_sc = find_all_trees_ds(c)
        bottom_scs.extend(bottom_sc)
        num_nodes += num_sub_nodes
        node_scs.extend(node_sc)
        for path in paths:
            return_tree.append([tree] + path)
    return return_tree, num_nodes, bottom_scs, node_scs


def convert_tree_to_ds(tree):
    '''
    Convert a tree to descriptors for the tree
    depth, number of linear trees, number of nodes,
    maximum number of branches, maximum sc, target sc, minimum sc,
    num of bottom compounds, maximum bottom sc, minimum bottom sc
    :param tree:
    :return:
    '''

    target_sc = tree['sc']

    trees, num_compounds, bottom_scs, node_scs = find_all_trees_ds(tree)

    num_bottom_compounds = len(trees)
    bottom_min_sc = min(bottom_scs)
    bottom_max_sc = max(bottom_scs)

    node_scs = node_scs[1:]            # first is the target
    node_min_sc = min(node_scs)
    node_max_sc = max(node_scs)
    num_nodes = len(node_scs)

    linear_trees = []
    for t in trees:
        if len([0 for c in t[-2]['child'] if c['child']]) == 0:
            if t[0:-1] not in [lt[0:-1] for lt in linear_trees]:
                linear_trees.append(t)

    depth = max([len(t) for t in linear_trees]) - 1
    num_linear_trees = len(linear_trees)

    return target_sc, depth, num_linear_trees, num_nodes, num_bottom_compounds, bottom_min_sc, \
           bottom_max_sc, node_min_sc, node_max_sc


def convert_one_record_to_ds(record):
    '''

    :param record:
    :return:
    '''

    pathways = [record['true_data']]
    pathways.extend(record['generated_paths'])

    patent_ID = record['true_data']['patentID']

    descriptors = []
    for i, pathway in enumerate(pathways):
        descriptors_tree = convert_tree_to_ds(
            pathway['tree'],
        )
        descriptors_tree = list(descriptors_tree)
        descriptors_tree.append(patent_ID)

        if i == 0:
            descriptors_tree.append(1)
        else:
            descriptors_tree.append(0)

        descriptors.append(descriptors_tree)

    return descriptors


def merge_ds_into_batch(batch_inputs):
    descriptors = []
    patent_ID = []
    num_trees = []
    for batch in batch_inputs:
        for input in batch:
            descriptors.append(input[:-2])
        patent_ID.append(batch[0][-2])
        num_trees.append(len(batch))

    return np.array(descriptors).astype('float32'), patent_ID, num_trees

