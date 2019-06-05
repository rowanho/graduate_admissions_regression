from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from model import build_model

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

model = build_model(training_data,16)
