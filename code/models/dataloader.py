import numpy as np
import pandas as pd

def load_data(path="../data/cora/", dataset="cora"):
    #features and category of papers
    raw_data = pd.read_csv(path+dataset+'.content',sep = '\t',header = None)
    
    order = raw_data.iloc[:,0]
    features =raw_data.iloc[:,1:-1]
    labels = pd.get_dummies(raw_data[1434]) #need to change the idx on other dataset

    #split train: valid: test = 8:1:1
    train_features = features.sample(frac = 0.8)
    train_features = train_features.sort_index()
    tmp_features = features[~features.index.isin(train_features.index)]
    valid_features = tmp_features.sample(frac = 0.5)
    valid_features = valid_features.sort_index()
    test_features = tmp_features[~tmp_features.index.isin(valid_features.index)]

    train_labels = labels[labels.index.isin(train_features.index)]
    valid_labels = labels[labels.index.isin(valid_features.index)]
    test_labels = labels[labels.index.isin(test_features.index)]

    train_order = order[order.index.isin(train_features.index)]
    valid_order = order[order.index.isin(valid_features.index)]
    test_order = order[order.index.isin(test_features.index)]

    # network
    raw_data_cites = pd.read_csv(path+dataset+'.cites',sep = '\t',header = None)

    num1, num2, num3 = train_features.shape[0], valid_features.shape[0], test_features.shape[0], 
    matrix1 = np.zeros((num1,num1))
    matrix2 = np.zeros((num2,num2))
    matrix3 = np.zeros((num3,num3))

    #replace the id of paper to ordered number
    a, b = list(range(num1)), train_order.values
    c = zip(b,a)
    map1 = dict(c)
    a, b = list(range(num2)), valid_order.values
    c = zip(b,a)
    map2 = dict(c) 
    a, b = list(range(num3)), test_order.values
    c = zip(b,a)
    map3 = dict(c)

    #adj matrix
    for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
        if i in train_order.values and j in train_order.values:
            x = map1[i] ; y = map1[j]  
            matrix1[x][y] = matrix1[y][x] = 1
        elif i in valid_order.values and j in valid_order.values:
            x = map2[i] ; y = map2[j]  
            matrix2[x][y] = matrix2[y][x] = 1
        elif i in test_order.values and j in test_order.values:
            x = map3[i] ; y = map3[j]  
            matrix3[x][y] = matrix3[y][x] = 1

    return train_features, valid_features, test_features, matrix1, matrix2, matrix3, train_labels, valid_labels, test_labels, map1, map2, map3