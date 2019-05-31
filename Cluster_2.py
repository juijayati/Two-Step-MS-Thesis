import numpy as np
import os
import random as rn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




np.random.seed(42)
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'

data_source_folder = 'DataSplit2(random)_700_50'
destination_folder = 'DataSplit2(random)_700_50'
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

tr_super_labels = np.load(os.path.join(data_source_folder, 'tr_super_labels.npy'))

labels = tr_super_labels.argmax(axis = 1)

print(labels)

print('################################################# Cluster 0 #######################################################')

itemindex = np.where(labels==0)


train_x_c0 = train_x[itemindex]

print(train_x_c0)

train_y_c0 = train_y[itemindex]

samples_per_class_c0 = train_y_c0.sum(axis = 0)

print(train_y_c0)

print(samples_per_class_c0)

np.save(os.path.join(destination_folder, 'tr_features_c0.npy'), train_x_c0)
np.save(os.path.join(destination_folder, 'tr_labels_c0.npy'), train_y_c0)

print('################################################# Cluster 1 #######################################################')

itemindex = np.where(labels==1)


train_x_c1 = train_x[itemindex]

print(train_x_c1)

train_y_c1 = train_y[itemindex]

samples_per_class_c1 = train_y_c1.sum(axis = 0)

print(train_y_c1)

print(samples_per_class_c1)

np.save(os.path.join(destination_folder, 'tr_features_c1.npy'), train_x_c1)
np.save(os.path.join(destination_folder, 'tr_labels_c1.npy'), train_y_c1)


print('################################################# Cluster 2 #######################################################')

itemindex = np.where(labels==2)


train_x_c2 = train_x[itemindex]

print(train_x_c2)

train_y_c2 = train_y[itemindex]

samples_per_class_c2 = train_y_c2.sum(axis = 0)

print(train_y_c2)

print(samples_per_class_c2)

np.save(os.path.join(destination_folder, 'tr_features_c2.npy'), train_x_c2)
np.save(os.path.join(destination_folder, 'tr_labels_c2.npy'), train_y_c2)


print('################################################### Test Cluster Prediction #################################################')

ts_super_labels = np.load(os.path.join(data_source_folder, 'ts_super_labels.npy'))

labels = ts_super_labels.argmax(axis = 1)

print(labels)

np.save(os.path.join(destination_folder, 'test_clusters.npy'), labels)

print('################################################# Cluster 0 #######################################################')

itemindex = np.where(labels==0)


test_x_c0 = test_x[itemindex]

print(test_x_c0)

test_y_c0 = test_y[itemindex]

samples_per_class_c0 = test_y_c0.sum(axis = 0)

print(test_y_c0)

print(samples_per_class_c0)

np.save(os.path.join(destination_folder, 'ts_features_c0.npy'), test_x_c0)
np.save(os.path.join(destination_folder, 'ts_labels_c0.npy'), test_y_c0)

print('################################################# Cluster 1 #######################################################')

itemindex = np.where(labels==1)


test_x_c1 = test_x[itemindex]

print(test_x_c1)

test_y_c1 = test_y[itemindex]

samples_per_class_c1 = test_y_c1.sum(axis = 0)

print(test_y_c1)

print(samples_per_class_c1)

np.save(os.path.join(destination_folder, 'ts_features_c1.npy'), test_x_c1)
np.save(os.path.join(destination_folder, 'ts_labels_c1.npy'), test_y_c1)


print('################################################# Cluster 2 #######################################################')

itemindex = np.where(labels==2)


test_x_c2 = test_x[itemindex]

print(test_x_c2)

test_y_c2 = test_y[itemindex]

samples_per_class_c2 = test_y_c2.sum(axis = 0)

print(test_y_c2)

print(samples_per_class_c2)

np.save(os.path.join(destination_folder, 'ts_features_c2.npy'), test_x_c2)
np.save(os.path.join(destination_folder, 'ts_labels_c2.npy'), test_y_c2)