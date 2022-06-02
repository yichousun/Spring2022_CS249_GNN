Group 10: Boosting GNN’s Generalization via Graph Data Augmentation

# **Class Imbalance Problem:**

The code for the class imbalance problem can be found in the folder "Class Imbalance Problem". It can run with or without a GPU.

## Dependencies:
- python3
- ipdb
- pytorch1.0
- network 2.4
- sklearn
- scipy

## Datasets:
2 datasets are used as follows.
1. Cora
2. BlogCatalog 

The datasets can be downloaded from [Link to download](https://drive.google.com/drive/folders/1rfIfRPG7IlzDMAYqQ25HOQmLBCHcECQx?usp=sharing). After downloading the datasets add them in the "data" folder.

## Architecture:
The detailed architecture can be referred from the report.

## Experimentation:
The Jupyter Notebook experiment.ipynb in the folder provides some examples on how we can run the code.
For example:
1. To run the code with only GCN, use the following command:

python main-onlygcn.py --imbalance --run_folder='runs/Run2_28_05_2022_cora_onlygcn' --dataset=cora --setting='recon_newG' --lr=0.01 --weight_decay=5e-4 --nhid=16 --dropout=0.5 --model=gcn
2. To run the code with GCN and oversampling:

python main.py --imbalance --run_folder='runs/Run3_28_05_2022_cora_oversampling' --dataset=cora --setting='upsampling' --lr=0.01 --weight_decay=5e-4 --nhid=16 --dropout=0.5 --model=gcn
3. To run the code with GCN and reweighting:

python main.py --imbalance --run_folder='runs/Run4_28_05_2022_cora_reweight' --dataset=cora --setting='reweight' --lr=0.01 --weight_decay=5e-4 --nhid=16 --dropout=0.5 --model=gcn
4. To run the code with GCN and GraphSMOTE:

python main.py --imbalance --run_folder='runs/Run1_28_05_2022_cora' --dataset=cora --setting='recon_newG' --lr=0.01 --weight_decay=5e-4 --nhid=16 --dropout=0.5 --model=gcn

Following are the meanings of the flags:
- imbalance: flag to introduce synthetic imbalance into the dataset
- run_folder: path to store the files generated via TensorBoard
- dataset: folder where the data is stored
- setting: which method to use for dealing with class imbalance
- lr: learning rate
- weight_decay: weight decay factor
- nhid: number of hidden dimensions in the graph convolution network
- dropout: dropout probability
- model: model for node classification. In our code, we only use GCN.

You can use same commands to run the codes on the BlogCatalog dataset by changing the dataset flag to BlogCatalog and removing --imbalance flag. Cora dataset is balanced. We synthetically make it imbalanced. BlogCatalog dataset is already imbalanced.
