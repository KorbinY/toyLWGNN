import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
import random
from collections import defaultdict

from math import sqrt
import datetime
import argparse
import os

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import  generate_pseudo_labels

def train_f_c(X_features, X_labels, Y_labels, matrix, _model, _opti, _cri):
    tmp_order = pd.DataFrame(range(X_labels.shape[0])).sample(frac=1)
    epoch_loss = 0
    X_middle = torch.zeros(size=(X_features.shape[0],512))

    for step in tmp_order.values:
        step = step.item()
        X_middle[step] =  _model(1, step, matrix, X_features, X_labels)
    for step in tmp_order.values:  
        step = step.item()
        _opti.zero_grad()
        output = _model(2, step, matrix, X_middle, X_labels)
        tmp_label = torch.tensor(np.array(Y_labels[step]), dtype=torch.float32)        
        loss = _cri(output, tmp_label.argmax().unsqueeze(0))
        return loss


def train_f_g(X_features,  Y_labels,  matrix, _model, _opti, _cri):
        _opti.zero_grad()
        output = _model(X_features, torch.tensor(matrix, dtype=torch.float32))
        loss = _cri(output, Y_labels.argmax(1))
        return loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='LW-GNN model')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--epochs_p', type=int, default=50, metavar='N', help='number of epochs to train pseudo label predictor')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    from models import dataloader
    train_features, valid_features, test_features, matrix1, matrix2, matrix3, train_labels, valid_labels, test_labels, map1, map2, map3 = dataloader.load_data()

    # generate the pseudo labels, while conserve the true ones
    true_train_labels, train_labels, train_confidential = generate_pseudo_labels.get_pseudo_labels(train_labels, train_features, args.epochs_p, device)

    #prepare the two models
    from models import hetero, homo
    f_c  = hetero.labelwisepassing(1433).to(device)
    optimizer_c = optim.Adam(f_c.parameters(), lr=args.lr)    
    f_g = homo.homopassing(1433,7)
    optimizer_g = optim.Adam(f_g.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    #bi-level optimization
    phi = 0.5    # initial relative-para between homo and hetero model
    
    X_features = torch.tensor(np.array(train_features), dtype=torch.float32).to(device)
    X_labels = torch.tensor(np.array(train_labels), dtype=torch.float32).to(device)
    Y_labels = torch.tensor(np.array(true_train_labels), dtype=torch.float32).to(device)

    V_features = torch.tensor(np.array(valid_features), dtype=torch.float32).to(device)
    V_labels = torch.tensor(np.array(valid_labels), dtype=torch.float32).to(device)

    while True: # need to add convergence (break-loop conditions)
        # outer iterationm, use valid data to update $\phi$
        loss1 = train_f_c(V_features, V_labels, V_labels, matrix2, f_c, optimizer_c, criterion)
        loss2 = train_f_g(V_features,  V_labels,  matrix2, f_g, optimizer_g, criterion)
        com_loss =  phi*loss1 + (1-phi)*loss2
        phi = phi - args.lr*first_order_appr(com_loss, phi)
        for T in range(2):  #inner iteration, T==2, use training data to update two models' paras
            loss1 = train_f_c(X_features, X_labels, X_labels, matrix2, f_c, optimizer_c, criterion)
            loss2 = train_f_g(X_features,  X_labels,  matrix2, f_g, optimizer_g, criterion)
            com_loss =  phi*loss1 + (1-phi)*loss2
            com_loss.backward()
            optimizer_c.step()
            optimizer_g.step()






    




if __name__ == "__main__":
    main()
