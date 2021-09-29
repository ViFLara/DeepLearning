import pandas as pd

predictors = pd.read_csv('entradas_breast.csv')
clas = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
predictors_training, test_predictors, training_class, test_class = train_test_split(predictors, clas, test_size=0.25)

import keras
from keras.models import Sequential # Modelo sequencial
from keras.layers import Dense 

classifier = Sequential() # Criação de nova rede neural
# units = (30 (entradas) + 1 (saída)) / 2 -> primeira camada oculta = 16
# activation (função de ativação)
# input_dim = elementos na camada de entrada (mesmo número de atributos)
# kernel_initializer = inicialização dos pesos
classifier.add(Dense(units=16, activation=('relu'),
                     kernel_initializer='random_uniform', input_dim=30)) 

# Criação de outra camada oculta:
classifier.add(Dense(units=16, activation=('relu'),
                     kernel_initializer='random_uniform')) 
classifier.add(Dense(units=16, activation=('relu'),
                     kernel_initializer='random_uniform')) 
classifier.add(Dense(units=16, activation=('relu'),
                     kernel_initializer='random_uniform')) 
classifier.add(Dense(units=16, activation=('relu'),
                     kernel_initializer='random_uniform')) 

# Camada de saída:
# sigmoide , pois retorna valor entre 0 e 1
classifier.add(Dense(units=1, activation=('sigmoid')))

#optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)

classifier.compile(optimizer='adam', loss=('binary_crossentropy'), metrics=['binary_accuracy']) # Configura o modelo para o treino

# batch_size=(10) -> calcula o erro para 10 registros e atualiza os pesos.
classifier.fit(predictors_training, training_class, batch_size=(10), epochs=100)

weights0 = classifier.layers[0].get_weights()
weights1 = classifier.layers[1].get_weights()
weights2 = classifier.layers[2].get_weights()
weights3 = classifier.layers[3].get_weights()
weights4 = classifier.layers[4].get_weights()
weights5 = classifier.layers[5].get_weights()

predictors = classifier.predict(test_predictors)
predictors = (predictors > 0.5) # para transformar em true e false

# Avaliando pelo sklearn:
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(test_class, predictors)
matrix = confusion_matrix(test_class, predictors)

# Avaliando pelo Keras:
result = classifier.evaluate(test_predictors, test_class) # test_class -> valores reais, test_predictors -> valores previstos

