import numpy as np
import os
import random as rn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering




np.random.seed(42)
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'

data_source_folder = 'DataSplit2(random)_700_50'
destination_folder = 'DataSplit2(random)_700_50_cluster'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
train_x = np.load(os.path.join(data_source_folder, 'tr_features.npy'))

#train_x = np.delete(train_x, np.s_[193::1], 1)

print(train_x)
#plt.scatter(train_x[:,15], train_x[0:,10]);
test_x = np.load(os.path.join(data_source_folder, 'ts_features.npy'))
#test_x = np.delete(test_x, np.s_[193::1], 1)

train_y = np.load(os.path.join(data_source_folder, 'tr_labels.npy'))
test_y = np.load(os.path.join(data_source_folder, 'ts_labels.npy'))

#kmeans_model = KMeans(n_clusters=6, random_state=1).fit(train_x)
#labels = kmeans_model.labels_


agglo_model = AgglomerativeClustering(n_clusters=4, )



print(labels)

unique, counts = np.unique(labels, return_counts=True)

print(dict(zip(unique, counts)))



