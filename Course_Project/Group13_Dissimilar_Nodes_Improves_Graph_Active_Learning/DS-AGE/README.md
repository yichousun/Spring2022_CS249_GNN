# DS-AGE 

This code is modified based on the AGE algorithm proposed in https://arxiv.org/abs/1705.05085

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

## Options

For this section, the available options for [your-dataset] includes "cora", "citeseer" and "pubmed".

The availale options for [your-active-learning-score] includes "baseline" (AGE), "f_similarity" (AGE+FDS), "s_similarity" (AGE+SDS), "e_similarity" (AGE+EDS), and "combined" (AGE+FDS+SDS).

The default value for [initial-number-of-labels] is 4.

Feel free to include your own datasets and your own score designs in the code.
