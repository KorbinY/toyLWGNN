import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_fp(features, label, _model, _opti, _crit, device):
    tmp_input = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    tmp_label = torch.tensor(np.array(label).argmax(1), dtype=torch.long)

    _opti.zero_grad()
    output = _model(tmp_input)
    loss = _crit(output, tmp_label)
    loss.backward()
    _opti.step()
    return loss.item()


def get_pseudo_labels(train_labels, train_features, epochs, device):
        # prepare the f_p
    from models import pseudo_predictor
    f_p = pseudo_predictor.labelpredictor(dim_in=1433, dim_out=7).to(device)
    optimizer_p = optim.Adam(f_p.parameters(), lr=0.01)
    criterion_p = nn.CrossEntropyLoss()

    true_train_labels = train_labels.copy()
    train_train_labels = train_labels.sample(frac=0.05)
    unknown_train_labels = train_labels[~train_labels.index.isin(train_train_labels.index)]

    #train the pseudo label predictor (without validation)
    print('start training the pseudo label predictor')
    for epo in range(epochs):
        tmp_train_labels = train_train_labels.sample(frac=1)
        epoch_loss = 0
        for step in range(tmp_train_labels.shape[0]//8):
            tmp_label = tmp_train_labels[step*8:step*8+8]
            tmp_features = train_features.loc[tmp_label.index, ]

            train_fp(tmp_features, tmp_label, f_p, optimizer_p, criterion_p, device)
    
    # then generare the pseudo labels for the $unknown$ nodes in training set
    # I also add the ‘confidential vector’ as mentioned in slides: the given labels have the confidential degree of 105\%, while pseudo labels have the confidential degree equal to their softmax number (0~1) respectively.
    f_p.eval()
    train_confidential = pd.DataFrame(np.ones(shape=(train_labels.shape[0], 1)), columns=[ 'confidential'], index=train_labels.index) + 0.05

    for step in range(unknown_train_labels.shape[0]//8+1):
        tmp_label = unknown_train_labels[step *
                                        8:min(step*8+8, unknown_train_labels.shape[0])]
        tmp_features = train_features.loc[tmp_label.index, ]
        tmp_input = torch.tensor(np.array(tmp_features), dtype=torch.float32)

        output = f_p(tmp_input)  # softmax-vector
        prediction = output.argmax(dim=1)
        confidential = F.softmax(dim=1, input=output).max(dim=1).values
        j = 0
        for line in tmp_label.index:
            train_labels.loc[train_labels.index == line] = [0]*7
            train_labels.iloc[train_labels.index == line, prediction[j].item()] = 1
            train_confidential.loc[train_labels.index == line] = confidential[j].item()
            j += 1 
    return true_train_labels, train_labels, train_confidential