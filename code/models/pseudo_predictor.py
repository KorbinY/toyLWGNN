import torch.nn as nn

# Pseudo Label Prediction $f_P$
# “Only a part of nodes are provided with labels Y”, I assume 25\% of the nodes in training set (= 20\% of the whole data set) are given with labels.
class labelpredictor(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(labelpredictor, self).__init__()
        self.mlp1 = nn.Linear(dim_in, 64, bias=False)
        self.mlp2 = nn.Linear(64, dim_out, bias=False)

    def forward(self, features):
        out = self.mlp1(features)
        out = self.mlp2(out)
        return out


