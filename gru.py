from DataProccess import DataProccess
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Input, Dropout, GRU, Flatten
import keras 
import numpy as np
from keras.optimizers import Adam

adam = Adam(lr=0.01)

# get and proccess data
data = DataProccess()
x_train, y_train, x_test, y_test = data.return_proccessed_data()

# reshape input to be [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

model = Sequential()
model.add(GRU(16, input_shape = (x_train.shape[1],x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(GRU(16, return_sequences=True))
model.add(Dropout(0.1))
model.add(GRU(16, return_sequences=False))
model.add(Dropout(0.1))

# binary
model.add(Dense(1))
model.add(Activation('sigmoid'))

# multiclass
# model.add(Dense(5, activation='softmax'))

model.summary()

#binary
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

#multiclass
# model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

# save the model
# model.save("/model.hdf5")

loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)

print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(x_test)
np.savetxt('predict.txt', np.transpose([y_test,y_pred]), fmt='%01d')