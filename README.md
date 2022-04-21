# Model
The codes are designed for the model LW-GNN_{GCN}, which is the variant of LW-GNN. It replace the GCNII with general GNN for homophilic graphs.

# Dataset
The folder "data" contains the homophilic dataset Cora. For this dataset, there are two files named "data.cites" and "graph.content".

* .content: Each line represents the text information of a vertex. [num_of_paper, text_features, type]
* .cites: The edgelist file of current social network.


## Run
Run the following command for training CANE:
    python3 run.py  -- batch_size 8 --lr 0.01 --epochs_p 50 --epochs 100# toyLWGNN

## Resource
Dai, Enyan, Zhimeng Guo, and Suhang Wang. "Label-Wise Message Passing Graph Neural Network on Heterophilic Graphs." arXiv preprint arXiv:2110.08128 (2021).
