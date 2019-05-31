import numpy as np
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import random as rn
from NeuralNetKeras import Classifier
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras import backend as K
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.cm as cm



#data_source_folder = 'DataSplit_all_all'
data_source_folder = 'DataSplit2'
destination_folder = 'DataSplit_700_50_cluster'
n_features = 281
n_classes = 12
frame_size = 200
n_clust = 6
n = [0.9, 0.91, 0.91, 0.9, 0.91]
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


def load_data_2(tr_sample_size_per_class,ts_sample_size_per_class):
    #total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class
    features_file = data_source_folder + '/features_200ms.npy'
    labels_file = data_source_folder + '/labels_200ms.npy'
    features = np.load(features_file)
    labels = np.load(labels_file)
    tr = np.concatenate((features, labels), axis = 1)
    print('Total Sample Shape : ', tr.shape)
    samples_per_class = np.sum(labels, axis=0).astype(int)
    print('Samples per class in total space: ', dict(zip(np.arange(labels.shape[1]),samples_per_class)))
    return tr


def partition_train_test(total, tr_sample_size_per_class, ts_sample_size_per_class):
    total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class
    n_classes = total.shape[1] - n_features
    #print('# Classes: ',n_classes)
    tr_new = np.zeros((0,total.shape[1]))
    ts_new = np.zeros((0,total.shape[1]))
    for i in range(n_classes):
        #if i!=8:
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
    #tr_new = np.concatenate((tr_new, total[np.where(total[:,n_features+8]==1.0)]), axis = 0)
    #ts_new = np.concatenate((ts_new, total[np.where(total[:,n_features+8]==1.0)]), axis = 0)

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
    divisor = int(5000 / frame_size)
    #divisor = 1
    for i in range(n_classes):
        #if i != 8:
            ts_indices_size = int(ts_sample_size_per_class/divisor)
            total_indices_size = int((total_sample_size_per_class - divisor)/divisor)
            ts_sample_indices_part = np.random.choice(total_indices_size, size=ts_indices_size, replace=False)
            #print(ts_sample_indices_part)
            ts_sample_indices_part = ts_sample_indices_part * divisor
            #print(ts_sample_indices_part)
            tr_sample_indices = np.arange(total_sample_size_per_class)
            ts_sample_indices = [0] * ts_indices_size * divisor
            for index in range(len(ts_sample_indices_part)):
                for k in range(divisor):
                    ts_sample_indices[index*divisor + k] = ts_sample_indices_part[index] + k
            tr_sample_indices = np.delete(tr_sample_indices, ts_sample_indices)
            #print(len(tr_sample_indices), len(ts_sample_indices))
            class_index_in_data = n_features + i
            class_sample = total[np.where(total[:,class_index_in_data]==1.0)]
            class_sample = class_sample[0:total_sample_size_per_class,:]
            #print(class_sample.shape)
            tr_new = np.concatenate((tr_new,class_sample[tr_sample_indices]), axis=0)
            ts_new = np.concatenate((ts_new,class_sample[ts_sample_indices]), axis=0)
            #print('Training Sample Size in Run ',tr_new.shape)
            #print('Testing sample size in Run ', ts_new.shape)
    #tr_new = np.concatenate((tr_new, total[np.where(total[:,n_features+8]==1.0)]), axis = 0)
    #ts_new = np.concatenate((ts_new, total[np.where(total[:,n_features+8]==1.0)]), axis = 0)

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

def cluster_classes_all(train_x, train_y, test_x, test_y,  n_clust):

    train_x = np.concatenate((train_x, test_x), axis=0)
    train_y = np.concatenate((train_y, test_y), axis=0)

    train_x = train_x[:, 0:40]
    #print(train_x.shape, train_y.shape)

    n_classes = train_y.shape[1]
    #train_y = total[:, n_features : total.shape[1]]
    n_classes = train_y.shape[1]
    print(train_x.shape, train_y.shape, n_classes)
    #calculate_purity(train_x, train_y)
    kmeans_model = KMeans(n_clusters=n_clust, random_state=1).fit(train_x)
    labels = kmeans_model.labels_
    silhouette_avg = silhouette_score(train_x, labels)
    print("For n_clusters =", n_clust,
          "The average silhouette_score is :", silhouette_avg)
    #Sum_of_squared_distances = kmeans_model.inertia_
    #print(Sum_of_squared_distances)

    #agglo_model = AgglomerativeClustering(n_clusters=n_clust).fit(train_x)
    #labels = agglo_model.labels_
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
        cluster_proportions = dict(zip(np.arange(n_classes), samples_per_class[i]))

    class_in_cluster = np.argmax(samples_per_class, axis=0)
    print('Classes in Clusters : ',class_in_cluster)
    print(samples_per_class)
    run_silhoutte_analysis_2(train_x)
    return class_in_cluster, labels, train_x, train_y, samples_per_class

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


def hierarchical_classifier(train_x, train_y, test_x, test_y, train_super_labels, test_super_labels):
    hierarch_nn = []
    nn = Classifier(train_x, train_super_labels)
    hierarch_nn.append(nn)
    #hierarch_nn[0].run_model(test_x, test_super_labels)

    for i in range(train_super_labels.shape[1]):
        itemindex_tr = np.where(train_super_labels[:, i] == 1.0)
        itemindex_ts = np.where(test_super_labels[:, i] == 1.0)
        nn = Classifier(train_x[itemindex_tr], train_y[itemindex_tr])
        hierarch_nn.append(nn)
        #print(i+1, itemindex_tr, itemindex_ts)
        #hierarch_nn[i+1].run_model(test_x[itemindex_ts], test_y[itemindex_ts])
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

def general_classifier(train_x, train_y, test_x, test_y):
    nn = Classifier(train_x, train_y)
    return nn.run_model(test_x, test_y)

def plot(true, gen, custom_x_labels):
    print()

def calculate_variance_before(labels, train_x, train_y):

    unique = np.unique(labels)
    n_clust = len(unique)
    var_bef = np.zeros(n_clust)
    for i in range(n_clust):
        itemindex = np.where(labels==i)
        train_x_i = train_x[itemindex]
        var_i = np.var(train_x_i, axis=1)
        var_bef[i] = int(np.linalg.norm(var_i))

    print('Variance(before) per cluster ',dict(zip(unique, var_bef)))


def calculate_variance_after(class_in_cluster, train_x, train_y):
    unique = np.unique(class_in_cluster)
    n_clust = len(unique)
    var_after = np.zeros(n_clust)
    for i in range(n_clust):
        classindex = np.where(class_in_cluster==i)
        #print(classindex)
        train_y_class = np.argmax(train_y, axis = 1)
        #print(len(train_y_class))
        itemindex =  np.nonzero(np.in1d(train_y_class,classindex))[0]
        #print(len(itemindex))
        train_x_i = train_x[itemindex]
        var_i = np.var(train_x_i, axis=1)
        var_after[i] = int(np.linalg.norm(var_i))
    print('Variance(after) per cluster ', dict(zip(unique, var_after)))

def calculate_distance_before(labels, train_x, train_y):
    unique = np.unique(labels)
    n_clust = len(unique)
    dist_bef = np.zeros((n_clust,n_clust))

    for i in range(n_clust):
        for j in range(n_clust):
            itemindex_1 = np.where(labels == i)
            train_x_i_1 = train_x[itemindex_1]
            itemindex_2 = np.where(labels==j)
            train_x_i_2 = train_x[itemindex_2]
            dist_bef[i][j] = int(cluster_distance(train_x_i_1, train_x_i_2))

    print("Distance Before : ",dist_bef)

def calculate_distance_after(class_in_cluster, train_x, train_y):
    unique = np.unique(class_in_cluster)
    n_clust = len(unique)
    dist_after = np.zeros((n_clust,n_clust))

    for i in range(n_clust):
        for j in range(n_clust):
            classindex_1 = np.where(class_in_cluster == i)
            # print(classindex)
            train_y_class = np.argmax(train_y, axis=1)
            # print(len(train_y_class))
            itemindex_1 = np.nonzero(np.in1d(train_y_class, classindex_1))[0]
            # print(len(itemindex))
            train_x_i_1 = train_x[itemindex_1]
            classindex_2 = np.where(class_in_cluster == j)
            itemindex_2 = np.nonzero(np.in1d(train_y_class, classindex_2))[0]
            # print(len(itemindex))
            train_x_i_2 = train_x[itemindex_2]
            dist_after[i][j] = int(cluster_distance(train_x_i_1, train_x_i_2))

    print("Distance After : ", dist_after)

def cluster_distance(train_x_1, train_x_2):
    dist = 0
    for i in range(train_x_1.shape[0]):
        for j in range(train_x_2.shape[0]):
            x1 = train_x_1[i,:]
            x2 = train_x_2[j,:]
            x3 = np.subtract(x1,x2)
            dist += np.sum(np.square(x3))

    dist = dist / (train_x_1.shape[0] * train_x_2.shape[0])
    print(dist)


    return dist


def calculate_F_5_score(class_in_cluster, cluster_labels, train_x, train_y, samples_per_class, n_cluster):
    F_5 = 0.0
    TP = FP = TN = FN = 0
    beta = 5
    N = np.sum(samples_per_class)
    print('Total Samples: ',N)
    total_pairs = N*(N-1)/2
    print('Total Pairs: ', total_pairs)
    print(class_in_cluster)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    TP_FP = np.sum(comb(counts,2,exact=False))
    print('TP_FP: ', TP_FP)
    TP_available_pairs = samples_per_class[samples_per_class >= 2]
    TP = np.sum(comb(TP_available_pairs,2,exact=False))
    FP = TP_FP - TP
    print('TP: %f , FP: %f' % (TP,FP))
    TN_FN = total_pairs - TP_FP
    print('TN_FN: ',TN_FN)
    for cur_cluster in range(n_cluster - 1):
        for cur_class in range(samples_per_class.shape[1]):
            FN += samples_per_class[cur_cluster,cur_class] * np.sum(samples_per_class[cur_cluster+1:n_cluster, cur_class])
    TN = TN_FN - FN
    print('TN: %d FN: %d '%(TN,FN))
    P = TP / (TP_FP)
    R = TP / (TP + FN)
    F_5 = (beta * beta + 1) * P * R / (beta * beta * P + R)
    RI = (TP + TN) / (TP + FP + FN + TN)
    F_error = (1 + beta * beta) * TP / ((1 + beta * beta) * TP + beta * beta * FN + FP)
    F = 2 * P * R / (P + R)
    print('P: %f , R: %f , F: %f, F_5: %f , RI: %f , F_2: %f' % (P, R, F, F_5, RI, F_error))
    return F_5

def run_silhoutte_analysis(train_x):
    range_n_clusters = [5, 6, 7, 8]
    for n_cluster in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.2, 1])

        ax1.set_ylim([0, train_x.shape[0] + (n_cluster + 1) * 10])
        clusterer = KMeans(n_clusters=n_cluster, random_state=10).fit(train_x)
        cluster_labels = clusterer.labels_

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(train_x, cluster_labels)
        print("For n_clusters =", n_cluster, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(train_x, cluster_labels)
        y_lower = 10
        for i in range(n_cluster):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_cluster)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples
        #ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster labels")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        #ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_cluster)

        ax2.scatter(train_x[:, 0], train_x[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        #ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on the Data Set "
                      "with Number_of_Clusters = %d" % n_cluster),
                     fontsize=14, fontweight='bold')
        name = ("Silhoutte for N_Clust = {}.png").format(n_cluster)
        #plt.savefig(os.path.join('DataSplit3(random)_700_50/Photos',name))
    plt.show()


def run_silhoutte_analysis_2(train_x):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.2, 1])

        ax1.set_ylim([0, train_x.shape[0] + (n_clust + 1) * 10])
        clusterer = KMeans(n_clusters=n_clust, random_state=1).fit(train_x)
        cluster_labels = clusterer.labels_

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(train_x, cluster_labels)
        print("For n_clusters =", n_clust, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(train_x, cluster_labels)
        y_lower = 10
        for i in range(n_clust):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clust)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10 # 10 for the 0 samples
        #ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster labels")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clust)

        ax2.scatter(train_x[:, 0], train_x[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        #ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st MFCC feature")
        ax2.set_ylabel("Feature space for 2nd MFCC feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on the Data Set "
                      "with Number_of_Clusters = %d" % n_clust),
                     fontsize=14, fontweight='bold')
        name = ("silhoutte_nclust_{}.png").format(n_clust)
        #plt.savefig(os.path.join('DataSplit3(random)_700_50/Photos',name))
        plt.show()

def calculate_purity(train_x, train_y):
    range_n_clust = [11]
    purity_km = np.zeros(len(range_n_clust))
    purity_agglo = np.zeros(len(range_n_clust))
    index = 0
    for n_cluster in range_n_clust:
        km = KMeans(n_clusters= n_cluster, random_state= 1).fit(train_x)
        km_labels = km.labels_
        #unique, counts = np.unique(km_labels, return_counts=True)
        #print('Class samples per cluster ', dict(zip(unique, counts)))
        samples_per_class_km = np.zeros((n_cluster, n_classes))
        for i in range(n_cluster):
            itemindex = np.where(km_labels == i)
            train_x_ci = train_x[itemindex]
            #print('###################################### Cluster %d ###################################' % i)
            train_y_ci = train_y[itemindex]
            samples_per_class_km[i] = train_y_ci.sum(axis=0)
            #print(dict(zip(np.arange(n_classes), samples_per_class_km[i])))
            cluster_proportions = dict(zip(np.arange(n_classes), samples_per_class_km[i]))

        class_in_cluster_km = np.argmax(samples_per_class_km, axis=0)
        print('Classes in Clusters KM : ', class_in_cluster_km)
        #print(samples_per_class_km)

        count = 0

        for i in range(n_cluster):
            itemindex = np.where(class_in_cluster_km ==i)
            count += np.sum(samples_per_class_km[i,itemindex])

        purity_km[index] = round(count/train_x.shape[0], 4)

        calculate_F_5_score(class_in_cluster_km, km_labels, train_x, train_y, samples_per_class_km, n_cluster)

        agglo = AgglomerativeClustering(n_clusters=n_cluster).fit(train_x)
        agglo_labels = agglo.labels_
        #unique, counts = np.unique(agglo_labels, return_counts=True)
        #print('Class samples per cluster K', dict(zip(unique, counts)))
        samples_per_class_agglo = np.zeros((n_cluster, n_classes))
        for i in range(n_cluster):
            itemindex = np.where(agglo_labels == i)
            train_x_ci = train_x[itemindex]
            #print('###################################### Cluster %d ###################################' % i)
            train_y_ci = train_y[itemindex]
            samples_per_class_agglo[i] = train_y_ci.sum(axis=0)
            #print(dict(zip(np.arange(n_classes), samples_per_class_km[i])))
            cluster_proportions = dict(zip(np.arange(n_classes), samples_per_class_agglo[i]))

        class_in_cluster_agglo = np.argmax(samples_per_class_agglo, axis=0)
        print('Classes in Clusters Agglo: ', class_in_cluster_agglo)
        count = 0

        for i in range(n_cluster):
            itemindex = np.where(class_in_cluster_agglo == i)
            #print(samples_per_class_agglo[i, itemindex])
            count += np.sum(samples_per_class_agglo[i, itemindex])
        print(count / train_x.shape[0])

        purity_agglo[index] = round(count / train_x.shape[0], 4)
        calculate_F_5_score(class_in_cluster_agglo, agglo_labels, train_x, train_y, samples_per_class_agglo, n_cluster)

        index += 1

    print('Purity KM : ', purity_km)
    print('Purity Agglo : ', purity_agglo)


def modify_train_test(train_x, test_x):
    start_indx = 180
    finish_indx = 193
    train_x_part1 = train_x[:,0:start_indx]
    train_x_part2 = train_x[:,finish_indx:train_x.shape[1]]
    test_x_part1 = test_x[:,0:start_indx]
    test_x_part2 = test_x[:,finish_indx:test_x.shape[1]]
    train_x_modified = np.concatenate((train_x_part1, train_x_part2), axis=1)
    test_x_modified = np.concatenate((test_x_part1, test_x_part2), axis=1)
    print("Train Test shape : ", train_x_modified.shape[1], test_x_modified.shape[1])
    return train_x_modified, test_x_modified

def modify_train_test_2(train_x, test_x):
    #mfcc
    train_x_modify = train_x[:,0:40]
    test_x_modify = test_x[:,0:40]
    #chroma
    #train_x_modify = np.hstack((train_x_modify,train_x[:,40:52]))
    #test_x_modify = np.hstack((test_x_modify,test_x[:,40:52]))
    #mel
    train_x_modify = np.hstack((train_x_modify, train_x[:, 52:180]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 52:180]))
    #contrast
    #train_x_modify = np.hstack((train_x_modify, train_x[:, 180:187]))
    #test_x_modify = np.hstack((test_x_modify, test_x[:, 180:187]))
    #tonnetz
    #train_x_modify = np.hstack((train_x_modify, train_x[:, 187:193]))
    #test_x_modify = np.hstack((test_x_modify, test_x[:, 187:193]))
    #zcc
    train_x_modify = np.hstack((train_x_modify, train_x[:, 193:211]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 193:211]))
    #vzcr
    train_x_modify = np.hstack((train_x_modify, train_x[:, 211:212]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 211:212]))
    #hzccr
    train_x_modify = np.hstack((train_x_modify, train_x[:, 212:213]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 212:213]))
    #rms
    train_x_modify = np.hstack((train_x_modify, train_x[:, 213:231]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 213:231]))
    #lef
    train_x_modify = np.hstack((train_x_modify, train_x[:, 231:232]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 231:232]))
    #ste
    train_x_modify = np.hstack((train_x_modify, train_x[:, 232:271]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 232:271]))
    #lster
    #train_x_modify = np.hstack((train_x_modify, train_x[:, 271:272]))
    #test_x_modify = np.hstack((test_x_modify, test_x[:, 271:272]))
    #vsfflux
    train_x_modify = np.hstack((train_x_modify, train_x[:, 272:273]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 272:273]))
    #vhoc
    #train_x_modify = np.hstack((train_x_modify, train_x[:, 273:277]))
    #test_x_modify = np.hstack((test_x_modify, test_x[:, 273:277]))
    #mhoc
    train_x_modify = np.hstack((train_x_modify, train_x[:, 277:281]))
    test_x_modify = np.hstack((test_x_modify, test_x[:, 277:281]))
    print("Train Test shape : ", train_x_modify.shape[1], test_x_modify.shape[1])
    return train_x_modify, test_x_modify



def main():
    initialize_random_seed()

    hierarch_accr_knn = []
    hierarch_accr_agglo = []
    gen_accr = []

    learning_rate = 0.0001

    for run in range(1):

        itr = 0

        tr_sample_size_per_class = 750 - 150
        ts_sample_size_per_class = 150
        total_sample_size_per_class = tr_sample_size_per_class + ts_sample_size_per_class

        #destination_folder = 'DataSplit3(random)_' + str(tr_sample_size_per_class) + '_' + str(ts_sample_size_per_class)
        destination_folder = 'DataSplit2'
        # print('Deastination Folder: ', destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        #tr_sample_size_per_class = 500
        hidden_unit_1_mult = 0.8
        accr_simple = 0.0
        accr_hier_knn = 0.0
        accr_hier_agglo = 0.0
        total = load_data_2(tr_sample_size_per_class, ts_sample_size_per_class)

        train_x, train_y, test_x, test_y = partition_train_test_2(total, tr_sample_size_per_class,ts_sample_size_per_class)

        class_in_cluster_knn, cluster_labels, train_cluster_x, train_cluster_y, samples_per_class = cluster_classes_all(train_x, train_y, test_x, test_y, n_clust)
        #train_x = train_x[:, 0:40]
        #test_x = test_x[:, 0:40]
        F_5 = calculate_F_5_score(class_in_cluster_knn, cluster_labels, train_cluster_x, train_cluster_y, samples_per_class, n_clust)
        #calculate_variance_before(cluster_labels, train_cluster_x, train_cluster_y)
        #calculate_variance_after(class_in_cluster_knn, train_cluster_x, train_cluster_y)
        #calculate_distance_before(cluster_labels, train_cluster_x, train_cluster_y)
        #calculate_distance_after(class_in_cluster_knn, train_cluster_x, train_cluster_y)
        '''train_x, test_x = modify_train_test(train_x, test_x)
        tr_super_labels, ts_super_labels = label_super_classes(train_x, train_y, test_x, test_y, class_in_cluster_knn)
        hierarch_nn_knn = hierarchical_classifier(train_x, train_y, test_x, test_y, tr_super_labels, ts_super_labels)
        h_accr_knn = evaluate_hierarchical_classifier(hierarch_nn_knn, test_x, test_y, ts_super_labels)
        g_accr = general_classifier(train_x, train_y, test_x, test_y)

        accr_simple += g_accr
        accr_hier_knn += h_accr_knn
        itr += 1

        for i in range(4):
            train_x, train_y, test_x, test_y = partition_train_test(total, tr_sample_size_per_class, ts_sample_size_per_class)
            train_x, test_x = modify_train_test_2(train_x, test_x)
            tr_super_labels, ts_super_labels = label_super_classes(train_x, train_y, test_x, test_y, class_in_cluster_knn)
            hierarch_nn_knn = hierarchical_classifier(train_x, train_y, test_x, test_y, tr_super_labels,
                                                      ts_super_labels)
            h_accr_knn = evaluate_hierarchical_classifier(hierarch_nn_knn, test_x, test_y, ts_super_labels)
            g_accr = general_classifier(train_x, train_y, test_x, test_y)

            accr_simple += g_accr
            accr_hier_knn += h_accr_knn

            itr += 1

            #train_x = np.load(os.path.join(destination_folder, 'tr_features.npy'))
            #train_y = np.load(os.path.join(destination_folder, 'tr_labels.npy'))

            #class_in_cluster_knn = cluster_classes(train_x, train_y, n_clust)


            #calculate_variance_before(cluster_labels, train_x, train_y)
            #calculate_variance_after(class_in_cluster_knn, train_x, train_y)
            #calculate_distance_before(cluster_labels, train_x, train_y)
            #calculate_distance_after(class_in_cluster_knn, train_x, train_y)
        gen_accr.append(round(accr_simple / itr, 4))
        hierarch_accr_knn.append(round(accr_hier_knn / itr, 4))

    print(gen_accr)
    print(hierarch_accr_knn)'''








if __name__ == '__main__': main()