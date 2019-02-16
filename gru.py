from DataProcess import DataProcess
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, GRU
import keras 
import numpy as np
from keras.utils import np_utils
import time



# get and process data
data = DataProcess()
# x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_multiclass()
x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_binary()


# reshape input to be [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# multiclass
# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)
# y_test_21=np_utils.to_categorical(y_test_21)


model = Sequential()
model.add(GRU(120, input_shape = (x_train.shape[1],x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))

model.add(GRU(120, return_sequences=True))
model.add(Dropout(0.1))

model.add(GRU(120, return_sequences=False))
model.add(Dropout(0.1))

# binary
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

# multiclass
# model.add(Dense(5))
# model.add(Activation('softmax'))

model.summary()

# optimizer
adam = Adam(lr=0.002)

#binary
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

#multiclass
# model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)

# save the model
# model.save("model.hdf5")


loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("--- %s seconds ---" % (time.time() - start))

y_pred = model.predict_classes(x_test)

print("Loss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
print("Anomaly in Test: ",np.count_nonzero(y_test, axis=0))
print("Anomaly in Prediction: ",np.count_nonzero(y_pred, axis=0))