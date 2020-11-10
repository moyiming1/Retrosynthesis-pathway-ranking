import pickle
import lzma

# Get ASKCOS data from the DOI given in the manuscript
# https://figshare.com/articles/dataset/ASKCOS_generated_retrosynthesis_pathway_data/13172504
data_path = 'ASKCOS_pathway_test_0.pkl.xz'

with lzma.open(data_path, 'rb') as f:
    askcos_data = pickle.load(f)







