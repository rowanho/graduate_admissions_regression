#builds the model
#dataset_length - for defining the input size of the first layer
#hidden_size  the size of the 2 hidden layers
#learning_rate - the learning rate of the optimizer

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

def build_model(dataset_length,hidden_size,learning_rate):
    model = keras.Sequential([
        keras.layers.Dense(hidden_size,activation=tf.nn.relu, input_shape = [dataset_length]),
        keras.layers.Dense(hidden_size, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

#implements k fold validation
#k - the number of partitions to use
#data - the data to split into the training and validation paritions
#model_params - tuple of parameters for build_model
def k_fold(k, data,labels,model_params,no_epochs=10,batch_size = 32):
    num_val = len(data)//k
    #np.random.shuffle(data)
    validation_scores = []
    histories = []
    for i in range(k):
        validation_data = data[i*num_val:(i+1)*num_val]
        validation_labels = labels[i*num_val:(i+1)*num_val]
        training_data = data[:i*num_val].add(data[(i+1)*num_val:],fill_value = 0)
        training_labels = labels[:i*num_val].add(labels[(i+1)*num_val:],fill_value =0)
        #print(training_data,training_labels)
        model  = build_model(*model_params)
        hist = model.fit(training_data,training_labels,epochs = no_epochs,batch_size = batch_size,verbose = 1,validation_data = (validation_data,validation_labels))

        val_score = model.evaluate(validation_data,validation_labels,verbose=0)
        validation_scores.append(val_score[1])
        histories.append(hist)
    return validation_scores,histories
