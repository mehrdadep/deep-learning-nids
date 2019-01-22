from DataProccess import DataProccess
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
import keras 
import numpy as np

data = DataProccess()

x_train, y_train, x_test, y_test = data.return_proccessed_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# print(x_train.shape)

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 41))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)
score = model.evaluate(x_test, y_test, batch_size=2000)

print(score)