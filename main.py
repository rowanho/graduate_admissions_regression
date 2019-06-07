import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

from graphs import plot_history

def normalize(data, norms):
    return (data - norms['mean'])/norms['std']

#loads and normalises the data
def get_data(datapath):
    TEST_PROPORTION = 0.3
    cols = ['Serial No.','GRE Score', 'TOEFL Score', 'University Rating',
            'SOP', 'LOR', 'CGPA', 'Research','Chance of Admit']

    csv_dataset = pd.read_csv(datapath, names=cols,sep=',',skiprows =1,skipinitialspace = True)

    #serial number is irrelevant
    csv_dataset.pop('Serial No.')

    #randomnly split into test and training data
    test_data = csv_dataset.sample(frac=TEST_PROPORTION,random_state=0)
    training_data = csv_dataset.drop(test_data.index)

    #separate out what we are trying to predict -  the chance of admission
    test_labels = test_data.pop('Chance of Admit')
    training_labels = training_data.pop('Chance of Admit')

    training_norms = training_data.describe()
    training_norms = training_norms.transpose()

    test_data_normalized = normalize(test_data,training_norms)
    training_data_normalized = normalize(training_data,training_norms)

    return  training_data_normalized, training_labels,test_data_normalized,test_labels

#builds the keras model object
#dataset_length - for defining the input size of the first layer
#hidden_size  the size of the 2 hidden layers
#learning_rate - the learning rate of the optimizer

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

#runs the k fold evaluation
#we can use the results of this to give a more balanced view of what changing parameters does to our model
def k_fold_validation(dataset,model_params,k,batch_size,no_epochs):
    data,labels = dataset
    num_val = len(data)//k
    mean_abs_errors = []
    histories = []
    for i in range(k):
        validation_data = data[i*num_val:(i+1)*num_val]
        validation_labels = labels[i*num_val:(i+1)*num_val]
        training_data = data[:i*num_val].add(data[(i+1)*num_val:],fill_value = 0)
        training_labels = labels[:i*num_val].add(labels[(i+1)*num_val:],fill_value =0)
        #print(training_data,training_labels)
        model  = build_model(*model_params)
        hist = model.fit(training_data,training_labels,epochs = no_epochs,batch_size = batch_size,verbose = 0,validation_data = (validation_data,validation_labels))
        val_score = model.evaluate(validation_data,validation_labels,verbose=0)
        mean_abs_errors.append(val_score[1])
        histories.append(hist)
    avg = np.mean(mean_abs_errors)
    print("Average mean absolute error across the k fold = {0:2.2f}%".format(avg*100))
    plot_history('k fold validation histories', histories)


# runs the the final model
def final_training(dataset,model_params,batch_size,no_epochs):
    training_data, training_labels, test_data, test_labels = dataset
    model = build_model(*model_params)
    hist = model.fit(training_data,training_labels,validation_data = (test_data,test_labels),verbose=0,epochs=no_epochs,batch_size = batch_size)
    loss,mean_abs_error,mean_squared_error = model.evaluate(test_data,test_labels,verbose=0)
    model.save('model.h5')
    print("Average mean absolute error on the test data = {0:2.2f}%".format(mean_abs_error*100))
    print("Average mean squared error across the test data = {0:2.2f}%".format(mean_squared_error*100))
    plot_history('Normal training history',[hist])


if __name__ == "__main__":

    training_data,training_labels, test_data, test_labels = get_data('graduate-admissions/Admission_Predict_Ver1.1.csv')
    #parameters for building the model
    LAYER_SIZE = 256
    LEARNING_RATE = 0.0001
    NO_EPOCHS = 30
    BATCH_SIZE = 30
    K = 4
    model_params = (len(training_data.keys()),LAYER_SIZE,LEARNING_RATE)

    k_fold_validation((training_data, training_labels),model_params,K,BATCH_SIZE,NO_EPOCHS)
    final_training((training_data,training_labels, test_data, test_labels), model_params,BATCH_SIZE,NO_EPOCHS)
