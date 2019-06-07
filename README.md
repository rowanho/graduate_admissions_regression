# graduate_admissions_regression
A regression problem using tensorflow/keras with a small dataset

Given parameters such as test scores, number of research projects and university ranking, aims to predict the probability
of a student making it onto a graduate program.

Dataset can be found at:
https://www.kaggle.com/mohansacharya/graduate-admissions - with credit to "Mohan S Acharya, Asfia Armaan, Aneeta S Antony :
A Comparison of Regression Models for Prediction of Graduate Admissions,
IEEE International Conference on Computational Intelligence in Data Science 2019"

## Dataset

There are 7 parameters in the dataset which can affect the chance of admission:

|Serial No. | GRE Score | TOEFL Score | University Rating | SOP | LOR | CGPA | Research | Chance of  Admit |
| --- | -------- | -------- | ---------- | ---------------- | ------- | --- | ------- | ------------ |
| 1 | 337       |118         |4                 |4.5 |4.5 |9.65 |1        |0.92          |
| 2 | 324       |107         |4                 |4   |4.5 |8.87 |1        |0.76          |
| ... | ... | ... | ... | ... | ... | ... | ... | ...|

## Results

I used k fold validation to help find good parameters for the model with the small amount of data available.

The model achieved around 0.6% mean squared error, and around 5-6% mean absolute error during testing.

History of training with k-fold validation with 4 folds,  over 30 epochs:
![alt text](https://github.com/rowanho/graduate_admissions_regression/blob/master/graphs/validation.png "validation")

History of final training over 30 epochs :
![alt text](https://github.com/rowanho/graduate_admissions_regression/blob/master/graphs/training.png "training")
