import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import math
from sklearn.metrics import accuracy_score
from keras import backend as K


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




class Classifier(object):

    batch_size = 512
    validation_split = 0.1
    verbose = 0


    def __init__(self, train_x, train_y, training_epochs = 10, hidden_units = 1, hidden_unit_1_mult = 0.8, hidden_unit_2_mult = 0.0, learning_rate = 0.01, save = False):


        ######################################## Main Constructor Method ############################################
        self.train_x = train_x
        self.train_y = train_y
        self.training_epochs = training_epochs
        self.hidden_units = hidden_units
        self.hidden_unit_1_mult = hidden_unit_1_mult
        self.hidden_unit_2_mult = hidden_unit_2_mult
        self.learning_rate = learning_rate
        self.save = save
        self.model = self.build_model()


    def build_model(self):
        ################################### For same random seed ###################################################
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        tf.set_random_seed(12345)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        ######################################### Main Model ###########################################

        n_dim = self.train_x.shape[1]
        n_classes = self.train_y.shape[1]
        self.n_hidden_units_one = math.ceil(self.hidden_unit_1_mult * n_dim)
        if self.hidden_units == 2:
            self.n_hidden_units_two = math.ceil(self.hidden_unit_2_mult * n_dim)
        self.sd = 1 / np.sqrt(n_dim)

        inputs = tf.keras.Input(shape=(n_dim,), name='features')
        x = tf.keras.layers.Dense(self.n_hidden_units_one, activation=tf.nn.sigmoid, name='dense_1')(inputs)
        #tf.keras.layers.Dropout(0.5),
        if self.hidden_units == 2:
            x = tf.keras.layers.Dense(self.n_hidden_units_two, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation= tf.nn.sigmoid, name='dense_2')(x)
            #tf.keras.layers.Dropout(0.5)

        outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
        #self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy','binary_crossentropy'])

        #self.history = self.model.fit(self.train_x, self.train_y, epochs=self.training_epochs,batch_size=self.batch_size,validation_split=self.validation_split,verbose=self.verbose)

        self.model.fit(self.train_x, self.train_y, epochs=self.training_epochs,batch_size=self.batch_size,validation_split=self.validation_split,verbose=self.verbose)
        return self.model


    def build_model_2(self):
        ################################### For same random seed ###################################################
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        tf.set_random_seed(12345)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        ######################################### Main Model ###########################################
        n_input = self.train_x.shape[0]
        n_dim = self.train_x.shape[1]
        n_classes = self.train_y.shape[1]
        self.n_hidden_units_one = math.ceil(self.hidden_unit_1_mult * n_dim)
        if self.hidden_units == 2:
            self.n_hidden_units_two = math.ceil(self.hidden_unit_2_mult * n_dim)
        self.sd = 1 / np.sqrt(n_dim)
        inputs = tf.keras.Input(shape=(n_dim,), name='features')

        x = tf.keras.layers.LSTM(self.n_hidden_units_one, input_shape=(self.train_x.shape[1:]), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu, return_sequences=False, name='dense_1')(inputs)
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')



        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

        #self.history = self.model.fit(self.train_x, self.train_y, epochs=self.training_epochs,batch_size=self.batch_size,validation_split=self.validation_split,verbose=self.verbose)

        self.model.fit(self.train_x, self.train_y, epochs=self.training_epochs,batch_size=self.batch_size,validation_split=self.validation_split,verbose=self.verbose)
        return self.model


    def update_model(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.training_epochs,batch_size=self.batch_size,validation_split=self.validation_split,verbose=self.verbose)


    def run_model(self, test_x, test_y):


        test_acc = self.model.evaluate(test_x, test_y)

        #print('Test accuracy:', test_acc)

        predictions = self.model.predict(test_x)

        #print(predictions[0])

        ##print(np.argmax(test_y[0]))

        y_true = test_y.argmax(axis=1)

        #print(predictions)
        y_predictions = predictions.argmax(axis=1)
        accr = accuracy_score(y_true, y_predictions)
        accr = round(accr, 4)
        print('Test Accuracy : ',accr)

        false_count = 0
        true_count = 0
        #for i in range(test_x.shape[0]):
        #    if np.argmax(predictions[i]) == np.argmax(test_y[i]):
        #       true_count += 1
        #       # print('true')
        #   else:
        #       false_count += 1
        #       # print('false')

        #print('True count : ', true_count)
        #print('False count : ', false_count)

        return accr

        # np.testing.assert_allclose(predictions, new_predictions, atol=1e-6)




















#model.save(os.path.join(destination_folder, model_name))





def main():
    initialize_random_seed()
    data_source_folder = 'DataSplit3(random)_700_50'
    destination_folder = 'DataSplit3(random)_700_50/Models'
    model_name = 'DataSplit3_3(random)_700_50_cluster_c2.h5'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    train_x = np.load(os.path.join(data_source_folder, 'tr_features.npy'))

    # train_x = np.delete(train_x, np.s_[193::1], 1)
    print(train_x.shape)
    # plt.scatter(train_x)
    test_x = np.load(os.path.join(data_source_folder, 'ts_features.npy'))
    # test_x = np.delete(test_x, np.s_[193::1], 1)

    train_y = np.load(os.path.join(data_source_folder, 'tr_labels_super.npy'))
    test_y = np.load(os.path.join(data_source_folder, 'ts_labels_super.npy'))

    nn = Classifier(train_x, train_y)


    #model = nn.build_model()

    nn.run_model(test_x, test_y)
    partition = 2
    test_x_pre = test_x[len(test_x)//partition:]
    test_x_post = test_x[:len(test_x)//partition]
    test_y_pre = test_y[len(test_y)//partition:]
    test_y_post = test_y[:len(test_y)//partition]

    nn.update_model(test_x_pre, test_y_pre)
    nn.run_model(test_x_post, test_y_post)

    #nn.model.save(os.path.join(destination_folder, model_name))





if __name__ == '__main__': main()