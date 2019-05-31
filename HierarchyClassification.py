import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import math
from sklearn.metrics import accuracy_score


np.random.seed(42)
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

data_source_folder = 'DataSplit_700_50_cluster'
models_folder = 'DataSplit_700_50_cluster/Models'
model_name_prefix = 'DataSplit_700_50_cluster'

test_x = np.load(os.path.join(data_source_folder, 'ts_features.npy'))
test_y = np.load(os.path.join(data_source_folder, 'ts_labels.npy'))
test_clusters = np.load(os.path.join(data_source_folder,'test_clusters.npy'))

num_clusters = len(np.unique(test_clusters))
print(num_clusters)

correct_prediction_count = 0


for i in range(num_clusters):

    indices = np.where(test_clusters==i)
    test_x_frac = test_x[indices]
    test_y_frac = test_y[indices]
    model_number = test_clusters[i]
    model_name = model_name_prefix + '_c{0}.h5'
    model_name = model_name.format(model_number)
    print(model_name)
    model = tf.keras.models.load_model(os.path.join(models_folder, model_name))
    print(test_x_frac.shape)
    arr = np.reshape(test_x_frac,(test_x_frac.shape[0],test_x_frac.shape[1]))
    print(arr.shape)
    #predictions = model.predict(np.reshape(test_x[i],(test_x.shape[1],1)))
    predictions = model.predict_on_batch(test_x_frac)
    y_true = test_y_frac.argmax(axis = 1)
    y_pred = predictions.argmax(axis = 1)
    correct_prediction_count += np.sum(y_true == y_pred)

print(correct_prediction_count)
