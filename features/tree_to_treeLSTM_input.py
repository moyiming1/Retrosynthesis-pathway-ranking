import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
import numpy as np
import torch

def print_tree(tree,level=0):
    if 'index' in tree.keys():
        print('--'*level + str(tree['index'])+'-'+tree['smiles'])

    for child in tree['child']:
        print_tree(child, level=level+1)

def calculate_evaluation_orders(adjacency_list, tree_size):
    '''Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.
    The TreeLSTM model requires node_order and edge_order to be passed into the model along
    with the node features and adjacency_list.  We pre-calculate these orders as a speed
    optimization.
    '''
    # print(type(adjacency_list))
    adjacency_list = np.array(adjacency_list)
    # print(adjacency_list.shape)
    node_ids = np.arange(tree_size, dtype=int)

    node_order = np.zeros(tree_size, dtype=int)
    unevaluated_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adjacency_list[:, 0]
    child_nodes = adjacency_list[:, 1]

    n = 0
    while unevaluated_nodes.any():
        # Find which child nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[child_nodes]

        # Find the parent nodes of unevaluated children
        unready_parents = parent_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of parents with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_parents)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order


def label_node_index(node, n=0):
    node['index'] = n
    for child in node['child']:
        n += 1
        n = label_node_index(child, n)
    return n


def gather_node_features(node, key, level=0):
    features = [node[key]]
    # print('--' * level + str(node['index']) + '-' + node['idx'])
    for child in node['child']:
        features.extend(gather_node_features(child, key, level=level+1))
    return features


def gather_adjacency_list(node):
    adjacency_list = []
    for child in node['child']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(gather_adjacency_list(child))
    return adjacency_list


def convert_reaction_to_fp(rsmi, psmi, fpsize=2048):
    rsmi = rsmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print("Cannot build reactant mol due to {}".format(e))
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                       radius=2,
                                                       nBits=fpsize,
                                                       useFeatures=False,
                                                       useChirality=True)
        fp = np.empty(fpsize, dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        print(rsmi)
        return

    rfp = fp

    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        print("Cannot build product mol due to {}".format(e))
        return

    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                       radius=2,
                                                       nBits=fpsize,
                                                       useFeatures=False,
                                                       useChirality=True)
        fp = np.empty(fpsize, dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)

    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return

    pfp = fp

    rxnfp = pfp - rfp
    return np.asarray(pfp), np.asarray(rxnfp)


def convert_tree_to_treefp(tree, fpsize=2048, n=0):
    # treefp = None
    if tree['child']:
        psmi = tree['smiles']
        rsmi = '.'.join([c['smiles'] for c in tree['child']])

        pfp, rxnfp = convert_reaction_to_fp(rsmi, psmi, fpsize=fpsize)

        treefp = {'pfp': pfp,
                  'rxnfp': rxnfp,
                  'index': n,
                  'child': []}
        tree['index'] = n
        for c in tree['child']:
            if c['child']:
                n += 1
                output, n = convert_tree_to_treefp(c, fpsize=fpsize, n=n)
                treefp['child'].append(output)

        return treefp, n


def convert_tree_to_singleinput(tree, fpsize=2048):
    treefp, _ = convert_tree_to_treefp(tree, fpsize=fpsize)
    # label_node_index(treefp)
    pfp = np.vstack(gather_node_features(treefp, 'pfp'))
    rxnfp = np.vstack(gather_node_features(treefp, 'rxnfp'))

    adjacency_list = gather_adjacency_list(treefp)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(pfp))

    return {
        'pfp': pfp,
        'rxnfp': rxnfp,
        'node_order': node_order,
        'adjacency_list': np.array(adjacency_list),
        'edge_order': edge_order,
        'num_nodes': len(pfp),
        'num_trees': 1
    }


def convert_one_record(record, fpsize=2048):
    # convert all single trees
    true_input = convert_tree_to_singleinput(record['true_data']['tree'], fpsize=fpsize)
    generated_input = [convert_tree_to_singleinput(line['tree'], fpsize=fpsize) for line in record['generated_paths']]

    pfp = np.vstack([true_input['pfp']] + [line['pfp'] for line in generated_input])
    rxnfp = np.vstack([true_input['rxnfp']] + [line['rxnfp'] for line in generated_input])

    node_order = np.hstack([true_input['node_order']] + [line['node_order'] for line in generated_input])
    edge_order = np.hstack([true_input['edge_order']] + [line['edge_order'] for line in generated_input])
    num_nodes = [true_input['num_nodes']] + [line['num_nodes'] for line in generated_input]
    # adjacency_list needs to add offset when concencate the trees
    adjacency_list = []
    offset = 0
    for n, a_l in zip(num_nodes, [true_input['adjacency_list']] + [line['adjacency_list'] for line in generated_input]):
        adjacency_list.append(a_l + offset)
        offset += n
    adjacency_list = np.vstack(adjacency_list)

    return {
        'pfp': pfp,
        'rxnfp': rxnfp,
        'node_order': node_order,
        'adjacency_list': adjacency_list,
        'edge_order': edge_order,
        'num_nodes': num_nodes,
        'num_trees': 1 + len(record['generated_paths']),
        'treeNo': record['true_data']['treeNo'],
        'patentID': record['true_data']['patentID'],
    }


def merge_into_batch(batch, to_tensor=False, device=torch.device('cpu')):
    batch_size = len(batch)
    if to_tensor:
        def process_output(input, dtype='float32'):
            dtype_dict = {'float32': torch.float32,
                          'int64': torch.int64}
            return torch.tensor(input, device=device, dtype=dtype_dict[dtype])
    else:
        def process_output(input, dtype=None):
            dtype_dict = {'float32': np.float32,
                          'int64': np.int64}
            return input.astype(dtype=dtype_dict[dtype])

    pfp = np.vstack([record['pfp'] for record in batch])
    rxnfp = np.vstack([record['rxnfp'] for record in batch])

    node_order = np.hstack([record['node_order'] for record in batch])
    edge_order = np.hstack([record['edge_order'] for record in batch])

    num_nodes = []

    if type(batch[0]['num_nodes']) is list:
        for record in batch: num_nodes += record['num_nodes']
    else:
        for record in batch: num_nodes.append(record['num_nodes'])
    num_trees = [record['num_trees'] for record in batch]
    # this is used to process adjacency_list
    record_num_nodes = [record['pfp'].shape[0] for record in batch]

    # adjacency_list needs to add offset when concancate the trees
    adjacency_list = []
    offset = 0
    for n, a_l in zip(record_num_nodes, [record['adjacency_list'] for record in batch]):
        adjacency_list.append(a_l + offset)
        offset += n
    adjacency_list = np.vstack(adjacency_list)
    # record_group = np.hstack(record_group)

    return {
        'pfp': process_output(pfp, dtype='float32'),
        'rxnfp': process_output(rxnfp, dtype='float32'),
        'node_order': process_output(node_order, dtype='int64'),
        'adjacency_list': process_output(adjacency_list, dtype='int64'),
        'edge_order': process_output(edge_order, dtype='int64'),
        'num_nodes': num_nodes,  # this doesn't need to be tensor
        'num_trees': num_trees,
        'batch_size': batch_size,
        # need to revise, not sure whether it needs to be a tensor yet
    }


def convert_multiple_records_into_batch(records, to_tensor=False, device=torch.device('cpu')):
    converted_data = [convert_one_record(r) for r in records]
    return merge_into_batch(converted_data, to_tensor=to_tensor, device=device)


if __name__ == '__main__':
    import pickle

    with open(project_path + '/data/pathway_train_example.pkl', 'rb') as f:
        data = []
        for _ in range(20):
            data.append(pickle.load(f))
    #%%
    tree1 = data[0]['true_data']['tree']
    tree2 = data[1]['true_data']['tree']

    output1 = convert_tree_to_singleinput(tree1, fpsize=2048)
    output2 = convert_tree_to_singleinput(tree2, fpsize=2048)
    batch = merge_into_batch([output1, output2])

