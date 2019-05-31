import numpy as np
import os
import random as rn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math


np.random.seed(42)
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'


data_source_folder = 'DataSplit3(random)_700_50'
destination_folder = 'DataSplit3(random)_700_50'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
train_x = np.load(os.path.join(data_source_folder, 'tr_features.npy'))
train_x = train_x[:,0:40]
train_y = np.load(os.path.join(data_source_folder, 'tr_labels.npy'))

print(train_x.shape)

n_class = train_y.shape[1]

tr = np.zeros((0,700,train_x.shape[1]))

print(tr.shape)


for i in range(train_y.shape[1]):
    temp = train_x[np.where(train_y[:,i]==1.0)]
    temp = temp[np.newaxis, :, :]
    print(temp.shape)
    tr = np.append(tr,temp, axis = 0)
    print(tr.shape)

n_clust = 4

kmeans_model = KMeans(n_clusters=n_clust, random_state=1).fit(tr)
labels = kmeans_model.labels_

print('Labels: ', labels)

unique, counts = np.unique(labels, return_counts=True)

print('Samples per Cluster: ',dict(zip(unique, counts)))
