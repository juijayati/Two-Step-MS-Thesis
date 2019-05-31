import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import math
from sklearn.metrics import accuracy_score






data_source_folder = 'DataSplit3(random)_700_50'
models_folder = 'DataSplit3(random)_700_50/Models'
model_name_prefix = 'DataSplit3(random)_700_50_cluster'

test_x = np.load(os.path.join(data_source_folder, 'ts_features.npy'))
test_y = np.load(os.path.join(data_source_folder, 'ts_labels.npy'))
test_y_super = np.load(os.path.join(data_source_folder, 'ts_labels_super.npy'))
#test_clusters = np.load(os.path.join(data_source_folder,'test_clusters.npy'))


model_super = tf.keras.models.load_model(os.path.join(models_folder, 'DataSplit3(random)_700_50_cluster_super.h5'))
predictions_super = model_super.predict(test_x)
y_pred_super = predictions_super.argmax(axis=1)

print(y_pred_super)

y_true_super = test_y_super.argmax(axis=1)

print(accuracy_score(y_true_super, y_pred_super))



num_clusters = len(np.unique(y_pred_super))
print(num_clusters)

correct_prediction_count = 0


for i in range(num_clusters):

    indices = np.where(y_pred_super==i)
    test_x_frac = test_x[indices]
    test_y_frac = test_y[indices]
    model_name = model_name_prefix + '_c{0}.h5'
    model_name = model_name.format(i)
    print(model_name)
    model = tf.keras.models.load_model(os.path.join(models_folder, model_name))
    print(test_x_frac)
    #predictions = model.predict(np.reshape(test_x[i],(test_x.shape[1],1)))
    predictions = model.predict(test_x_frac)
    y_true = test_y_frac.argmax(axis = 1)
    y_pred = predictions.argmax(axis = 1)
    correct_prediction_count += np.sum(y_true == y_pred)

print(correct_prediction_count / test_x.shape[0])


