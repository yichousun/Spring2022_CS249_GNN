# DS-AGE 

This code is modified based on the AGE algorithm

@article{DBLP:journals/corr/CaiZC17,
  author    = {HongYun Cai and
               Vincent Wenchen Zheng and
               Kevin Chen{-}Chuan Chang},
  title     = {Active Learning for Graph Embedding},
  journal   = {CoRR},
  volume    = {abs/1705.05085},
  year      = {2017},
  url       = {https://arxiv.org/abs/1705.05085},
  timestamp = {Mon, 15 May 2017 06:49:04 GMT}
}

## Dependencies

networkx==2.6.3

numpy==1.21.6

scikit_learn==1.1.1

scipy==1.4.1

tensorflow==2.8.2

## Usage

```# In the algcn folder
python train_entropy_density_graphcentral_ts.py [your-active-learning-score] [initial-number-of-labels] [number-of-classes] [your-dataset]
```
