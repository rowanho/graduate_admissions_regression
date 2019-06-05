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
def k_fold(k, data,labels,model_params):
    num_val = len(data)//k
    #np.random.shuffle(data)
    validation_scores = []
    for i in range(k):
        validation_data = data[i*num_val:(i+1)*num_val]
        validation_labels = labels[i*num_val:(i+1)*num_val]
        training_data = data[:i*num_val].add(data[(i+1)*num_val:],fill_value = 0)
        training_labels = labels[:i*num_val].add(labels[(i+1)*num_val:],fill_value =0)
        #print(training_data,training_labels)
        model  = build_model(*model_params)
        hist = model.fit(training_data,training_labels,epochs = 10,verbose = 0)

        val_score = model.evaluate(validation_data,validation_labels,verbose=0)
        print(val_score)
        validation_scores.append(val_score)
    return validation_scores
