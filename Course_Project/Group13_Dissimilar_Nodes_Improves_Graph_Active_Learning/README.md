# Dissimilar Nodes Improve Graph Active Learning

## Project Introduction

In this paper, we introduce a new concept, dissimilarity, and propose three scoring functions based on this idea: feature dissimilarity score (FDS), structure dissimilarity score (SDS), and embedding dissimilarity score (EDS).

Our experimental results shows that our proposed method could boost the performance of node classification tasks of Graph Convolutional Networks by about 2.5$\%$ when the number of labels is fixed.

We also provide an ablation study to show that our methods are generalizable to many other GNN variants.

We examine the effectiveness of our AL scoring functions on heterophilic datasets.

## Descriptions

Folder DS-AGE is our main algorithm, which incorporate our dissimilarity scores into the active learning framework based on GCN backbone.

Folder model_variants includes implementation of other GNN-variants which is adaptable for our main algorithm.

Folder AGE-heterophily includes experiments on heterophilic datasets.

Folder KMetriods includes ablation study about the choose of clustering algorithms.

Folder experimental_snapshots includes the .ipynb files which records our experimental results.

Document presentation is our presentation slides.

And Document report is the general report for this project.
