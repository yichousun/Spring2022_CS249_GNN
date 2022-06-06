# CS249 RLogic Attention

This project aims to learn logical rules from a given knowledge base. RLogic model proposed here is a recursive model to learn the rules consisting body and head, and apply the same to deduce the head relation for unknown paths.
On top of the base RLogic model, this repo has 3 implementations:

* RLogic with naive attention: Uses naive dot product based attention to calculate head probabilites.
* RLogic with transformer based attention: Uses transformer based (key,query vectors) attention to determine the head relation probabilites.
* RLogic with GNN: Uses GNN to learn relation embeddings instead of LSTM (in base RLogic model).

As the code is a part of an ongoing research project, the repository is made private. Please email us to get access to the repository. 

Link to the code - [link](https://github.com/npochhi/CS249-RLogic-Attention/)
