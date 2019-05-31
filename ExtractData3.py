import numpy as np
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
import random as rn
from NeuralNetKeras import Classifier
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras import backend as K
import matplotlib.pyplot as plt


data_source_folder = 'DataSplit_all_all'
destination_folder = 'DataSplit_700_50_cluster'
n_features = 281
n_classes = 12
n_clust = 4

gen = [0.9, 0.91, 0.91, 0.9, 0.91]
hier = [0.94, 0.95, 0.95, 0.94, 0.94]


def initialize_random_seed():
    np.random.seed(12345)
    rn.seed(12345)
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    tf.set_random_seed(12345)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def save_data(destination_folder, train_x, train_y, test_x, test_y, train_super, test_super, n_input = 5):
    np.save(os.path.join(destination_folder,'tr_features.npy'),train_x)
    np.save(os.path.join(destination_folder,'ts_features.npy'),test_x)
    np.save(os.path.join(destination_folder,'tr_labels.npy'),train_y)
    np.save(os.path.join(destination_folder,'ts_labels.npy'),test_y)
    if n_input != 5:
        np.save(os.path.join(destination_folder, 'tr_labels_super.npy'), train_super)
        np.save(os.path.join(destination_folder, 'ts_labels_super.npy'), test_super)
        for i in range(train_super.shape[1]):
            itemindex_tr = np.where(train_super[:,i]==1.0)
            itemindex_ts = np.where(test_super[:,i]==1.0)
            np.save(os.path.join(destination_folder,'tr_features_c{0}.npy'.format(i)),train_x[itemindex_tr])
            np.save(os.path.join(destination_folder, 'tr_labels_c{0}.npy'.format(i)), train_y[itemindex_tr])
            np.save(os.path.join(destination_folder, 'ts_features_c{0}.npy'.format(i)), test_x[itemindex_ts])
            np.save(os.path.join(destination_folder, 'ts_labels_c{0}.npy'.format(i)), test_y[itemindex_ts])


def load_data(tr_sample_size_per_class,ts_sample_size_per_class):
    total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class
    tr_features_file = data_source_folder + '/tr_features.npy'
    tr_labels_file = data_source_folder + '/tr_labels.npy'
    ts_features_file = data_source_folder + '/ts_features.npy'
    ts_labels_file = data_source_folder + '/ts_labels.npy'
    tr_features = np.load(tr_features_file)
    tr_labels = np.load(tr_labels_file)
    ts_features = np.load(ts_features_file)
    ts_labels = np.load(ts_labels_file)
    tr_features = np.concatenate((tr_features,ts_features), axis = 0)
    tr_labels = np.concatenate((tr_labels, ts_labels), axis = 0)
    tr = np.concatenate((tr_features,tr_labels), axis = 1)
    #print('Total Sample Shape : ', tr.shape)
    samples_per_class = np.sum(tr_labels, axis=0).astype(int)
    #print('Samples per class in total space: ', dict(zip(np.arange(tr_labels.shape[1]),samples_per_class)))
    itemindex = np.where(tr[:,(tr_features.shape[1]+13)]==1)
    tr = np.delete(tr, itemindex, 0)
    tr = np.delete(tr, (tr_features.shape[1]+13), 1)
    #print('Sample Shape after deleting Class 14: ', tr.shape)
    itemindex = np.where(tr[:,(tr_features.shape[1]+3)]==1)
    tr = np.delete(tr,itemindex,0)
    tr = np.delete(tr, (tr_features.shape[1]+3), 1)
    #print('Sample Shape after deleting Class 4 and 14: ', tr.shape)
    return tr



def partition_train_test(total, tr_sample_size_per_class, ts_sample_size_per_class):
    total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class
    n_classes = total.shape[1] - n_features
    #print('# Classes: ',n_classes)
    tr_new = np.zeros((0,total.shape[1]))
    ts_new = np.zeros((0,total.shape[1]))
    for i in range(n_classes):
        ts_sample_indices = np.random.choice(total_sample_size_per_class, size=ts_sample_size_per_class, replace=False)
        tr_sample_indices = np.arange(total_sample_size_per_class)
        tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)
        class_index_in_data = n_features + i
        class_sample = total[np.where(total[:,class_index_in_data]==1.0)]
        class_sample = class_sample[0:total_sample_size_per_class,:]
        #print(class_sample.shape)
        tr_new = np.concatenate((tr_new,class_sample[tr_sample_indices]), axis=0)
        ts_new = np.concatenate((ts_new,class_sample[ts_sample_indices]), axis=0)
        #print('Training Sample Size in Run ',tr_new.shape)
        #print('Testing sample size in Run ', ts_new.shape)
    tr_new_features = tr_new[:,0:n_features]
    tr_new_labels = tr_new[:,n_features:(n_features+n_classes)]
    ts_new_features = ts_new[:,0:n_features]
    ts_new_labels = ts_new[:,n_features:(n_features+n_classes)]
    return tr_new_features, tr_new_labels, ts_new_features, ts_new_labels



def partition_train_test_2(total, tr_sample_size_per_class, ts_sample_size_per_class):
    total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class
    n_classes = total.shape[1] - n_features
    #print('# Classes: ',n_classes)
    tr_new = np.zeros((0,total.shape[1]))
    ts_new = np.zeros((0,total.shape[1]))
    for i in range(n_classes):
        ts_indices_size = int(ts_sample_size_per_class/5)
        total_indices_size = total_sample_size_per_class-5
        ts_sample_indices_part = np.random.choice(total_indices_size, size=ts_indices_size, replace=False)
        #print(ts_sample_indices_part)
        tr_sample_indices = np.arange(total_sample_size_per_class)
        ts_sample_indices = [0] * ts_indices_size * 5
        for index in range(len(ts_sample_indices_part)):
            for k in range(5):
                ts_sample_indices[index*5 + k] = ts_sample_indices_part[index] + k
        tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)
        class_index_in_data = n_features + i
        class_sample = total[np.where(total[:,class_index_in_data]==1.0)]
        class_sample = class_sample[0:total_sample_size_per_class,:]
        #print(class_sample.shape)
        tr_new = np.concatenate((tr_new,class_sample[tr_sample_indices]), axis=0)
        ts_new = np.concatenate((ts_new,class_sample[ts_sample_indices]), axis=0)
        #print('Training Sample Size in Run ',tr_new.shape)
        #print('Testing sample size in Run ', ts_new.shape)
    tr_new_features = tr_new[:,0:n_features]
    tr_new_labels = tr_new[:,n_features:(n_features+n_classes)]
    ts_new_features = ts_new[:,0:n_features]
    ts_new_labels = ts_new[:,n_features:(n_features+n_classes)]
    return tr_new_features, tr_new_labels, ts_new_features, ts_new_labels



def cluster_classes(train_x, train_y, n_clust):
    #print(train_x.shape, train_y.shape)
    train_x = train_x[:, 0:40]
    n_classes = train_y.shape[1]
    kmeans_model = KMeans(n_clusters=n_clust, random_state=1).fit(train_x)
    labels = kmeans_model.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print('Class samples per cluster ',dict(zip(unique, counts)))
    samples_per_class = np.zeros((n_clust, n_classes))
    for i in range(n_clust):
        itemindex = np.where(labels == i)
        train_x_ci = train_x[itemindex]
        print('###################################### Cluster %d ###################################' % i)
        train_y_ci = train_y[itemindex]
        samples_per_class[i] = train_y_ci.sum(axis=0)
        print(dict(zip(np.arange(n_classes), samples_per_class[i])))

    class_in_cluster = np.argmax(samples_per_class, axis=0)

    print('Classes in Clusters : ',class_in_cluster)

    return class_in_cluster

def cluster_classes_all(train_x, train_y, test_x, test_y, n_clust):
    #print(train_x.shape, train_y.shape)
    train_x = np.concatenate((train_x, test_x), axis = 0)
    train_y = np.concatenate((train_y, test_y), axis = 0)
    train_x = train_x[:, 0:40]
    n_classes = train_y.shape[1]
    #kmeans_model = KMeans(n_clusters=n_clust, random_state=1).fit(train_x)
    #labels = kmeans_model.labels_
    agglo_model = AgglomerativeClustering(n_clusters=n_clust).fit(train_x)
    labels = agglo_model.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print('Class samples per cluster ',dict(zip(unique, counts)))
    samples_per_class = np.zeros((n_clust, n_classes))
    for i in range(n_clust):
        itemindex = np.where(labels == i)
        train_x_ci = train_x[itemindex]
        print('###################################### Cluster %d ###################################' % i)
        train_y_ci = train_y[itemindex]
        samples_per_class[i] = train_y_ci.sum(axis=0)
        print(dict(zip(np.arange(n_classes), samples_per_class[i])))

    class_in_cluster = np.argmax(samples_per_class, axis=0)

    print('Classes in Clusters : ',class_in_cluster)

    return class_in_cluster

def cluster_classes_hier(train_x, train_y, n_clust):
    train_x = train_x[:, 0:40]
    n_classes = train_y.shape[1]
    agglo_model = AgglomerativeClustering(n_clusters=n_clust).fit(train_x)
    labels = agglo_model.labels_

    # kmeans_model = KMeans(n_clusters=n_clust, random_state=1).fit(train_x)
    # labels = kmeans_model.labels_
    unique, counts = np.unique(labels, return_counts=True)
    # print('Class samples per cluster ',dict(zip(unique, counts)))
    samples_per_class = np.zeros((n_clust, n_classes))
    for i in range(n_clust):
        itemindex = np.where(labels == i)
        train_x_ci = train_x[itemindex]
        # print('###################################### Cluster %d ###################################' % i)
        train_y_ci = train_y[itemindex]
        samples_per_class[i] = train_y_ci.sum(axis=0)
        # print(dict(zip(np.arange(n_classes), samples_per_class[i])))

    class_in_cluster = np.argmax(samples_per_class, axis=0)

    # print('Classes in Clusters : ',class_in_cluster)

    return class_in_cluster



def label_super_classes(train_x, train_y, test_x, test_y, class_in_cluster):
    tr_super_labels = np.zeros((train_y.shape[0],(np.max(class_in_cluster)+1)))
    ts_super_labels = np.zeros((test_y.shape[0],(np.max(class_in_cluster)+1)))

    for i in range(len(class_in_cluster)):
        cluster = class_in_cluster[i]
        itemindex = np.where(train_y[:,i]==1)
        tr_super_labels[itemindex,cluster] = 1.0
        itemindex = np.where(test_y[:,i]==1)
        ts_super_labels[itemindex,cluster] = 1.0
    return tr_super_labels, ts_super_labels


def hierarchical_classifier(train_x, train_y, test_x, test_y, train_super_labels, test_super_labels, learning_rate):
    hierarch_nn = []
    nn = Classifier(train_x, train_super_labels, learning_rate = learning_rate)
    hierarch_nn.append(nn)
    hierarch_nn[0].run_model(test_x, test_super_labels)

    for i in range(train_super_labels.shape[1]):
        itemindex_tr = np.where(train_super_labels[:, i] == 1.0)
        itemindex_ts = np.where(test_super_labels[:, i] == 1.0)
        nn = Classifier(train_x[itemindex_tr], train_y[itemindex_tr], learning_rate = learning_rate)
        hierarch_nn.append(nn)
        hierarch_nn[i+1].run_model(test_x[itemindex_ts], test_y[itemindex_ts])
    return hierarch_nn


def evaluate_hierarchical_classifier(hierarch_nn, test_x, test_y, test_super_labels):
    predictions_super = hierarch_nn[0].model.predict(test_x)
    y_pred_super = predictions_super.argmax(axis=1)
    y_true_super = test_super_labels.argmax(axis=1)
    print('Super Level Accuracy : ',accuracy_score(y_true_super, y_pred_super))
    num_clusters = len(np.unique(y_pred_super))
    #print("Clusters : ",num_clusters)

    clusters = np.unique(y_pred_super)

    correct_prediction_count = 0

    for i in clusters:
        indices = np.where(y_pred_super == i)
        test_x_frac = test_x[indices]
        test_y_frac = test_y[indices]
        predictions = hierarch_nn[i+1].model.predict(test_x_frac)
        y_true = test_y_frac.argmax(axis=1)
        y_pred = predictions.argmax(axis=1)
        correct_prediction_count += np.sum(y_true == y_pred)

    accr = round(correct_prediction_count / test_x.shape[0], 4)

    print('Hierarchical Classification Accuracy : ',accr)
    return accr

def evaluate_hierarchical_classifier_dyn(hierarch_nn, test_x, test_y, test_super_labels):
    predictions_super = hierarch_nn[0].model.predict(test_x)
    y_pred_super = predictions_super.argmax(axis=1)
    y_true_super = test_super_labels.argmax(axis=1)
    print('Super Level Accuracy : ',accuracy_score(y_true_super, y_pred_super))
    num_clusters = len(np.unique(y_pred_super))
    #print("Clusters : ",num_clusters)

    clusters = np.unique(y_pred_super)

    correct_prediction_count = 0

    for i in clusters:
        indices = np.where(y_pred_super == i)
        test_x_frac = test_x[indices]
        test_y_frac = test_y[indices]
        predictions = hierarch_nn[i+1].model.predict(test_x_frac)
        y_true = test_y_frac.argmax(axis=1)
        y_pred = predictions.argmax(axis=1)
        correct_prediction_count += np.sum(y_true == y_pred)

    accr = round(correct_prediction_count / test_x.shape[0], 4)

    print('Hierarchical Classification Accuracy : ',accr)
    return accr

def general_classifier(train_x, train_y, test_x, test_y, learning_rate):
    nn = Classifier(train_x, train_y, learning_rate = learning_rate)
    return nn.run_model(test_x, test_y)

def plot(true, gen, custom_x_labels):
    print()


def main():
    initialize_random_seed()

    hierarch_accr_knn = []
    hierarch_accr_agglo = []
    gen_accr = []

    learning_rate = 0.0001

    for run in range(1):

        tr_sample_size_per_class = 700
        ts_sample_size_per_class = 50
        total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class

        destination_folder = 'DataSplit3(random)_' + str(tr_sample_size_per_class) + '_' + str(
            ts_sample_size_per_class)
        # print('Deastination Folder: ', destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        #tr_sample_size_per_class = 500
        hidden_unit_1_mult = 0.8
        accr_simple = 0.0
        accr_hier_knn = 0.0
        accr_hier_agglo = 0.0
        for i in range(1):


            total = load_data(tr_sample_size_per_class, ts_sample_size_per_class)
            train_x, train_y, test_x, test_y = partition_train_test(total, tr_sample_size_per_class,
                                                                    ts_sample_size_per_class)

            train_x = np.load(os.path.join(destination_folder, 'tr_features.npy'))
            train_y = np.load(os.path.join(destination_folder, 'tr_labels.npy'))

            #class_in_cluster_knn = cluster_classes(train_x, train_y, n_clust)
            class_in_cluster_knn = cluster_classes_all(train_x, train_y, test_x, test_y, n_clust)
            #print(class_in_cluster_knn)
            tr_super_labels, ts_super_labels = label_super_classes(train_x, train_y, test_x, test_y, class_in_cluster_knn)

            #save_data(destination_folder, train_x, train_y, test_x, test_y, tr_super_labels, ts_super_labels, 7)

            hierarch_nn_knn = hierarchical_classifier(train_x, train_y, test_x, test_y, tr_super_labels, ts_super_labels, learning_rate)
            h_accr_knn = evaluate_hierarchical_classifier(hierarch_nn_knn, test_x, test_y, ts_super_labels)
            #hierarch_accr.append(h_accr_knn)

            #class_in_cluster_hier = cluster_classes_hier(train_x, train_y, n_clust)
            # print(class_in_cluster_hier)
            #tr_super_labels, ts_super_labels = label_super_classes(train_x, train_y, test_x, test_y, class_in_cluster_hier)
            #hierarch_nn_agglo = hierarchical_classifier(train_x, train_y, test_x, test_y, tr_super_labels,
             #                                         ts_super_labels, hidden_unit_1_mult)
            #h_accr_aggglo = evaluate_hierarchical_classifier(hierarch_nn_agglo, test_x, test_y, ts_super_labels)

            g_accr = general_classifier(train_x, train_y, test_x, test_y, learning_rate)

            accr_simple += g_accr
            accr_hier_knn += h_accr_knn
            #accr_hier_agglo += h_accr_aggglo

        learning_rate = learning_rate * 10
        gen_accr.append(accr_simple / 1.0)
        hierarch_accr_knn.append(accr_hier_knn / 1.0)
        #hierarch_accr_agglo.append(accr_hier_agglo / 1.0)

    print(gen_accr)
    print(hierarch_accr_knn)
    #print(hierarch_accr_agglo)








if __name__ == '__main__': main()