import tensorflow as tf
from tensorflow import keras

import numpy as np

import pandas as pd

from model import build_model, k_fold

from graphs import plot_history
DATAPATH = 'graduate-admissions/Admission_Predict_Ver1.1.csv'
TEST_PROPORTION = 0.3
cols = ['Serial No.','GRE Score', 'TOEFL Score', 'University Rating',
        'SOP', 'LOR', 'CGPA', 'Research','Chance of Admit']

csv_dataset = pd.read_csv(DATAPATH, names=cols,sep=',',skiprows =1,skipinitialspace = True)

#serial number is irrelevant
csv_dataset.pop('Serial No.')

#randomnly split into test and training data
test_data = csv_dataset.sample(frac=TEST_PROPORTION,random_state=0)
training_data = csv_dataset.drop(test_data.index)

#separate out what we are trying to predict -  the chance of admission
test_labels = test_data.pop('Chance of Admit')
training_labels = training_data.pop('Chance of Admit')

#normalize the data
def normalize(data, norms):
    return (data - norms['mean'])/norms['std']

training_norms = training_data.describe()
training_norms = training_norms.transpose()


test_data_normalized = normalize(test_data,training_norms)
training_data_normalized = normalize(training_data,training_norms)

#parameters for building the model
#includes the learning rate and the size of the two hidden layers
LAYER_SIZE = 128
LEARNING_RATE = 0.0001
NO_EPOCHS = 30
BATCH_SIZE = 20
model_params = (len(training_data.keys()),LAYER_SIZE,LEARNING_RATE)

#k fold evaluation
mean_abs_errors,histories = k_fold(4, training_data_normalized, training_labels, model_params,no_epochs=NO_EPOCHS,batch_size=BATCH_SIZE)

avg = np.mean(mean_abs_errors)


#the final model
model = build_model(*model_params)

hist = model.fit(training_data_normalized,training_labels,validation_data = (test_data_normalized,test_labels),verbose=1,epochs=NO_EPOCHS,batch_size = BATCH_SIZE)

loss,mean_abs_error,mean_squared_error = model.evaluate(test_data_normalized,test_labels,verbose=1)

print("Average mean absolute error across the k fold = {0:2.2f}%".format(avg*100))

print("Average mean absolute error on the test data = {0:2.2f}%".format(mean_abs_error*100))

print("Average mean squared error across the test data = {0:2.2f}%".format(mean_squared_error*100))
plot_history('k fold validation histories', histories)

plot_history([hist])
