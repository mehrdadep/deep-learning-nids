from DataProccess import DataProccess
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Input, Dropout, GRU, Flatten,SimpleRNN
import keras 
import numpy as np
from keras.utils import np_utils

adam = Adam(lr=0.01)

# get and proccess data
data = DataProccess()
x_train, y_train, x_test, y_test = data.return_proccessed_data()

# reshape input to be [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])


model = Sequential()
model.add(SimpleRNN(x_train.shape[1], input_shape = (x_train.shape[1],x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(x_train.shape[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(x_train.shape[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(x_train.shape[1], return_sequences=False))
model.add(Dropout(0.2))

# binary
model.add(Dense(1,activation='sigmoid'))

# multiclass
# model.add(Dense(5, activation='softmax'))

model.summary()

#binary
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

#multiclass
# model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# save the model
# model.save("/model.hdf5")

loss, accuracy = model.evaluate(x_test, y_test)

print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# y_pred = model.predict_classes(x_test)
# np.savetxt('predict.txt', np.transpose([y_test,y_pred]), fmt='%01d')