from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

DATAPATH = 'graduate-admissions/Admission_Predict_Ver1.1.csv'
TEST_AMOUNT = 200
cols = ['Serial No.','GRE Score', 'TOEFL Score', 'University Rating',
        'SOP', 'LOR', 'CGPA', 'Research','Chance of Admit']
csv_dataset = pd.read_csv(DATAPATH, names=cols,sep=',',skipinitialspace = True)

#serial number is irrelevant
csv_dataset.pop('Serial No.')


test_data = csv_dataset[:TEST_AMOUNT]

training_data = csv_dataset[TEST_AMOUNT:]
