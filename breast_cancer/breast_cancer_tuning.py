import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

import tensorflow.compat.v2 as tf
from tensorflow.keras import backend as k 

from keras.optimizer_v1 import Optimizer
from keras.optimizer_v2 import adam as adam_v2
from keras.optimizer_v2 import adamax as adamax_v2
from keras.optimizer_v2 import optimizer_v2

predictors = pd.read_csv('entradas_breast.csv')
clas = pd.read_csv('saidas_breast.csv')

def create_network(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer, input_dim=30)) 
    classifier.add(Dropout(0.2))
    # Criação de outras camadas ocultas:
    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer)) 
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer)) 
    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer)) 
    classifier.add(Dense(units=neurons, activation=activation,
                         kernel_initializer=kernel_initializer))  

    # Camada de saída:
    classifier.add(Dense(units=1, activation=('sigmoid')))
    
    # Configuração do modelo para o treino
    classifier.compile(optimizer=optimizer, loss=loss,
                       metrics=['binary_accuracy']) 
    
    return classifier

k.clear_session()

classifier = KerasClassifier(build_fn=(create_network))

parameters = {'batch_size': [10,30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring=('accuracy'),
                           cv=5)
grid_search.fit(predictors, clas)

best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_



