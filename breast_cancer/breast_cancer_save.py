import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

# Salvando no formato json
classifier_json = classifier.to_json()
with open('classifier_breast.json', 'w') as json_file:
    # Salvando em disco
    json_file.write(classifier_json)
    
# Salvando os pesos
classifier.save_weights('classifier_breast.h5')
