import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import backend as k 

from keras.optimizer_v1 import Optimizer
from keras.optimizer_v2 import adam as adam_v2
from keras.optimizer_v2 import adamax as adamax_v2
from keras.optimizer_v2 import optimizer_v2

predictors = pd.read_csv('entradas_breast.csv')
clas = pd.read_csv('saidas_breast.csv')

classifier = Sequential()
classifier.add(Dense(units=8, activation='relu',
                         kernel_initializer='normal', input_dim=30)) 
classifier.add(Dropout(0.2))
# Criação de outras camadas ocultas:
classifier.add(Dense(units=8, activation='relu',
                         kernel_initializer='normal')) 
classifier.add(Dropout(0.2))
classifier.add(Dense(units=8, activation='relu',
                         kernel_initializer='normal')) 
# Camada de saída:
classifier.add(Dense(units=1, activation=('sigmoid')))    
# Configuração do modelo para o treino
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['binary_accuracy']) 

classifier.fit(predictors, clas, batch_size=10, epochs=100)

new = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

prediction = classifier.predict(new)
prediction = (prediction > 0.9)

    