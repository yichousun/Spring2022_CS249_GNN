# Model Variants

This directory is for different backbones used for learning stage of active learning. In total we tested 4 models, GCN, GAT, GCNII and SGC.

## Usage

All following procedures should be run using conda virtual environment.

### GCN

```bash
conda env create -f age_gcn.yml
conda activate age_gcn
# in repo gcn
python setup.py install
# in repo AGE/algcn
# refer to https://github.com/vwz/AGE for more instructions
# example model usage follows:
python original.py 0 4 6 citeseer
```

### GAT

```bash
conda env create -f age_gat.yml
conda activate age_gat
# in repo GAT
# example model usage follows:
python gat.py cora
```

### SGC

```bash
conda env create -f age_sgc.yml
conda activate age_sgc
# in repo SGC
# example model usage follows:
python citation.py cora
```

### GCNII

```bash
# can be run with SGC environment
conda activate age_sgc
# in repo GCNII
# example model usage follows:
python train.py --data cora
```