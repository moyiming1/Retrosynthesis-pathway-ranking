# Retrosynthesis-pathway-ranking

## Installation

### Requirements:

- Python >= 3.7
- PyTorch >= 1.4
- RDKit >= 2019.09
- numpy >= 1.17
- scikit-learn >= 0.22.1 
- hdbscan >= 0.8.26

### Data:
Example data are supplied in the "/data" folder. 

Trained model is in "/trained_model/treeLSTM256-fp2048.pt".

### Code:

- Run training
```
python train_treeLSTM.py
```

- Run testing
```
python test_treeLSTM.py
```

- A wrapper for pathway ranking model is in "pathway_ranker.py"

- Code for extracting pathways from single-step reaction database is in "pathway_extraction/extract.py"