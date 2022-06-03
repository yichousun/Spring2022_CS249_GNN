## Learn_to_Hash_GNN_Recommender_System

This is the Pytorch implementation for our CS 249 course project.

## Introduction

In this work, we proposed a new framework called HashLGN which consists of two parts: a LightGCN encoder for learning node representations, and a hash and decoding function to achieve higher efficiency in inner product search. 
Our model can achieve comparable performance with the state-of-art models but with much higher efficiency.



## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide the training and testing data set for the MovieLens 1M data set, which include ratings of 6040 users and 1000209 user-item interactions in total. We choose 80 percent of each user's historical interactions at random to form the training set for each dataset, and the rest is treated as the test set.

## How to run 

run LightGCN on **ml-1m** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command for training using the top k maximum inner product search method

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ml-1m" --topks="[20]" --recdim=64`

* command for training using the top k maximum inner product search method

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ml-1m" --threhold=5 --recdim=64`

* log output

```shell
...
======================
EPOCH[5/1000]
BPR[sample time][16.2=15.84+0.42]
[saved][[BPR[aver loss1.128e-01]]
[0;30;43m[TEST][0m
{'precision': array([0.0299514]), 'recall': array([0.93652087]), 'ndcg': array([0.8482945]), 'f1': array([0.05799485])}
[TOTAL TIME] 35.9975962638855
...
======================
EPOCH[116/1000]
BPR[sample time][16.9=16.60+0.45]
[saved][[BPR[aver loss2.056e-02]]
[TOTAL TIME] 30.99874997138977
...
```


## Extend:
* If you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.
* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.
* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`

