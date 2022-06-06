# Model Variants

This directory is for different backbones used for learning stage of active learning. In total we tested 4 models, GCN, GAT, GCNII and SGC.

## Usage

All following procedures should be run using conda virtual environment.

### GCN

Refer to the folder DS-AGE

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
