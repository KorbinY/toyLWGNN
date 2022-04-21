import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

# f_G
class homopassing(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(homopassing, self).__init__()

        self.trans1 = nn.Linear(dim_in, dim_in,bias=False)
        self.trans2 = nn.Linear(dim_in, dim_in//2,bias=False)
        self.trans3 = nn.Linear(dim_in//2, dim_out,bias=False)
        
    def forward(self, X, A):
        X = F.relu(self.trans1(A.mm(X)))
        X = F.relu(self.trans2(A.mm(X)))
        return self.trans3(A.mm(X))