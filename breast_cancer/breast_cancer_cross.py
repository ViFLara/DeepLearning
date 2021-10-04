import pandas as pd
import keras
import tensorflow.compat.v2 as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras import backend
from keras.optimizer_v1 import Optimizer
from keras.optimizer_v2 import adam as adam_v2
from keras.optimizer_v2 import optimizer_v2

predictors = pd.read_csv('entradas_breast.csv')
clas = pd.read_csv('saidas_breast.csv')

def create_network():
    classifier = Sequential()
    classifier.add(Dense(units=16, activation=('relu'),
                         kernel_initializer='random_uniform', input_dim=30)) 
    classifier.add(Dropout(0.2))
    # Criação de outras camadas ocultas:
    classifier.add(Dense(units=16, activation=('relu'),
                         kernel_initializer='random_uniform')) 
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=16, activation=('relu'),
                         kernel_initializer='random_uniform')) 
    classifier.add(Dense(units=16, activation=('relu'),
                         kernel_initializer='random_uniform')) 
    classifier.add(Dense(units=16, activation=('relu'),
                         kernel_initializer='random_uniform')) 

    # Camada de saída:
    classifier.add(Dense(units=1, activation=('sigmoid')))
    
    # Configuração do modelo para o treino
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)
    classifier.compile(optimizer=optimizer, loss=('binary_crossentropy'),
                       metrics=['binary_accuracy']) 
    
    return classifier

classifier = KerasClassifier(build_fn=create_network, epochs=100, batch_size=10)
results = cross_val_score(estimator=classifier, X=predictors, y=clas,
                          cv=10, scoring='accuracy')

mean = results.mean()

# Verificando quantos valores estão desviando da média:
deviation = results.std()