import numpy as np
import os
import random as rn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import math
from sklearn.metrics import davies_bouldin_score

from sklearn.mixture import GaussianMixture




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

entropy_intra_cluster = np.zeros(12)

entropy_inter_cluster = np.ones(12)

for i_clust in range(4, 5):

    print('######################################## Number of Clusters %d ########################################### ' % i_clust)
    n_clust = 6
    n_classes = train_y.shape[1]


    #kmeans_model = KMeans(n_clusters=n_clust, random_state=1).fit(train_x)
    #labels = kmeans_model.labels_

    agglo_model = AgglomerativeClustering(n_clusters=n_clust).fit(train_x)
    labels = agglo_model.labels_


    #print('Labels: ', labels)

    unique, counts = np.unique(labels, return_counts=True)
    samples_per_class = np.zeros((n_clust, n_classes))


    print('Samples per Cluster: ',dict(zip(unique, counts)))

    for i in range(n_clust):
        itemindex = np.where(labels == i)
        train_x_ci = train_x[itemindex]
        print('###################################### Cluster %d ###################################' % i)
        train_y_ci = train_y[itemindex]
        samples_per_class[i] = train_y_ci.sum(axis=0)
        print(dict(zip(np.arange(n_classes), samples_per_class[i])))

    class_in_cluster = np.argmax(samples_per_class, axis=0)



    print('Classes in Clusters : ', class_in_cluster)

    break