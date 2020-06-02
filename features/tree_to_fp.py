import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
# from utils.visualize_tree import create_tree_html
import numpy as np
import scipy

def find_all_trees(tree):
    if len(tree['child']) == 0:
        return [[tree]]
    return [[tree] + path for c in tree['child'] for path in find_all_trees(c)]

def linearize_tree_artifical(tree, return_sc=False):
    trees = find_all_trees(tree)
    linear_tree = []
    for t in trees:
        if len([0 for c in t[-2]['child'] if c['child']]) == 0:
            if t[0:-1] not in [lt[0:-1] for lt in linear_tree]:
                linear_tree.append(t)

    simple_linear_tree = []
    for lt in linear_tree:
        simple_lt = []
        other = []
        for i, node in enumerate(lt):
            if i == len(lt) - 1:
                # last step there is no main smiles
                simple_lt.append({'main': '',
                                  'other': other})
            else:
                if return_sc:
                    simple_lt.append({'main': node['sc'],
                                      'other': [o for o in other if o != node['sc']]})
                    other = [n['sc'] for n in node['child']]
                else:
                    simple_lt.append({'main': node['smiles'],
                                      'other': [o for o in other if o != node['smiles']]})
                    other = [n['smiles'] for n in node['child']]

        simple_linear_tree.append(simple_lt)
    return simple_linear_tree, len(simple_linear_tree)

def linearize_tree_true(tree):
    trees = find_all_trees(tree)
    linear_tree = []
    for t in trees:
        if len([0 for c in t[-2]['child'] if c['child']]) == 0:
            if t[0:-1] not in [lt[0:-1] for lt in linear_tree]:
                linear_tree.append(t)

    simple_linear_tree = []
    for lt in linear_tree:
        simple_lt = []
        other = []
        for i, node in enumerate(lt):
            if i == len(lt) - 1:
                # last step there is no main smiles
                simple_lt.append({'main': '',
                                  'namerxn': 'leaf',
                                  'other': other})
            else:
                simple_lt.append({'main': node['smiles'],
                                  'namerxn': node['namerxn'],
                                  'other': [o for o in other if o != node['smiles']]})
                other = [n['smiles'] for n in node['child']]
        simple_linear_tree.append(simple_lt)
    return simple_linear_tree, len(simple_linear_tree)

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


# use this
def convert_tree_to_fp(tree, fpsize=2048, max_step=10):
    linear_trees, num_linear_trees = linearize_tree_artifical(tree)

    tree_pfp = np.zeros((num_linear_trees, max_step, fpsize))
    tree_rxnfp = np.zeros((num_linear_trees, max_step, fpsize))
    for k, lt in enumerate(linear_trees):
        for i in range(len(lt)-1):
            psmi = lt[i]['main']
            if lt[i+1]['main']:
                rsmi = '.'.join(lt[i+1]['other'] + [lt[i+1]['main']])
            else:
                rsmi = '.'.join(lt[i + 1]['other'])
            tree_pfp[k, i, :], tree_rxnfp[k, i, :] = convert_reaction_to_fp(rsmi, psmi, fpsize=fpsize)

    return tree_pfp, tree_rxnfp, num_linear_trees


def convert_tree_to_smiles(tree):
    linear_trees, num_linear_trees = linearize_tree_artifical(tree)

    smiles = set()
    for k, lt in enumerate(linear_trees):
        for i in range(len(lt) - 1):
            psmi = lt[i]['main']
            smiles.add(psmi)
            if lt[i + 1]['main']:
                rsmi = lt[i + 1]['other'] + [lt[i + 1]['main']]
            else:
                rsmi = lt[i + 1]['other']

            smiles.update(set(rsmi))

    return smiles

def convert_tree_to_sc(tree, max_step=12, scscore_model=None, scscore_dict=None):
    linear_trees, num_linear_trees = linearize_tree_artifical(tree)

    tree_psc = np.zeros((num_linear_trees, max_step))
    tree_rsc = np.zeros((num_linear_trees, max_step))

    for k, lt in enumerate(linear_trees):
        for i in range(len(lt)-1):
            psmi = lt[i]['main']
            if lt[i+1]['main']:
                rsmi = lt[i+1]['other'] + [lt[i+1]['main']]
            else:
                rsmi = lt[i + 1]['other']

            if scscore_model:
                tree_psc[k, i] = scscore_model.get_score_from_smi(psmi)[1]
            elif scscore_dict:
                tree_psc[k, i] = scscore_dict[psmi]

            rsc = -np.Inf
            for smi in rsmi:
                if scscore_model:
                    rsc_temp = scscore_model.get_score_from_smi(smi)[1]
                elif scscore_dict:
                    rsc_temp = scscore_dict[smi]
                if rsc_temp > rsc:
                    rsc = rsc_temp

            tree_rsc[k, i] = rsc

    return tree_psc, tree_rsc, num_linear_trees


def convert_tree_reaction_class(tree, class_dict, max_step=10):
    linear_trees, num_linear_trees = linearize_tree_true(tree)

    class_index = []
    forward_class_index = []
    backward_class_index = []
    for i, lt in enumerate(linear_trees):

        depth = len(lt) - 1
        # print('depth: ', depth)
        for j, reaction in enumerate(lt):
            if reaction['namerxn'] != 'leaf':
                class_idx = class_dict[reaction['namerxn']]
                # print(class_idx)
                class_index.append([i, j, class_idx])
                if j != 0:
                    forward_class_index.append([i, j-1, class_idx])
                if j != depth - 1:
                    backward_class_index.append([i, max_step-j-2, class_idx])

    return class_index, forward_class_index, backward_class_index

if __name__ == '__main__':
    import pickle
    with open('/home/yiming/Projects/data/pathway_ranking/training_trees/pathway_train_data_0.pkl', 'rb') as f:
        data = pickle.load(f)

    with open(project_path + '/data/reaction_class_dict.pkl', 'rb') as f:
        class_dict = pickle.load(f)
    #%%
    for i in range(len(data)):
        tree = data[i]['true_data']['tree']
        # create_tree_html([tree], 'test1')
        # a = find_all_trees(tree)
        a = linearize_tree_true(tree)
        # b = convert_tree_to_fp(tree, fpsize=2048)
        c, d, e = convert_tree_reaction_class(tree, class_dict, max_step=10)


