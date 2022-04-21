import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# then we aggregate the message in a label-wise way.
# f_C
class labelwisepassing(nn.Module):
    def __init__(self, dim_in):
        super(labelwisepassing, self).__init__()
        self.trans1 = nn.Linear(dim_in, 64)
        self.trans2 = nn.Linear(512, 64)
        self.relu = nn.ReLU()
        self.pooling = torch.nn.MaxPool1d((2)) # assume num of layers = 2
        self.predict = nn.Linear(512,7)

    def forward(self, flag, index, matrix, x_features, x_labels):
        if flag == 1:
            z = self.trans1(x_features)
        else:
            z = self.trans2(x_features)
    
        indices = matrix[index].nonzero() # index of classes exists in a node's neighbors
        para = 1/np.sqrt(matrix[indices].sum(1)*sum(matrix[index]))
        para = torch.tensor(para, dtype=torch.float32).unsqueeze(0)

        tmp_a = torch.zeros(size=(7,64))
        if len(x_labels[indices]) != 0: #jump loop if no neighbor
            for i in range(7):
                if sum(np.array(x_labels[indices]))[i] == 0:
                    continue #use the zero embedding for empty class i
                else:
                    neighbor = np.nonzero(np.array(x_labels[indices])[:,i])
                    tmp_a[i] = sum(para[0][neighbor].unsqueeze(1)*z[indices][neighbor])
        h = torch.cat((z[index].unsqueeze(0), tmp_a.view(1,-1)), 1)
        h = self.relu(h)

        if flag == 1:
            return h
        else:
            h_combi = torch.stack((x_features[index].unsqueeze(0), h))
            h_combi = self.pooling(h_combi.squeeze().permute(1,0))
            out = self.predict(h_combi.permute(1,0))
            return out