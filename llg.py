from DataProcess import DataProcess
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Input, Concatenate, GRU, Conv1D,Flatten
import keras 
import numpy as np
from keras.utils import np_utils


# get and process data
data = DataProcess()
# x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_multiclass()
x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_binary()


# make 3 inputs (each 1/3 of row size)
sliceCount1 = int(x_train.shape[1]/3)
sliceCount2 = sliceCount1 + sliceCount1

x_train_1 = x_train[:,0:sliceCount1]
x_train_2 = x_train[:,sliceCount1:sliceCount2]
x_train_3 = x_train[:,sliceCount2:]

x_test_1 = x_test[:,0:sliceCount1]
x_test_2 = x_test[:,sliceCount1:sliceCount2]
x_test_3 = x_test[:,sliceCount2:]

# reshape input to be [samples, timesteps, features]
# x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
# x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

x_train_1 = x_train_1.reshape(x_train_1.shape[0], 1, x_train_1.shape[1])
x_train_2 = x_train_2.reshape(x_train_2.shape[0], 1, x_train_2.shape[1])
x_train_3 = x_train_3.reshape(x_train_3.shape[0], 1, x_train_3.shape[1])
x_test_1 = x_test_1.reshape(x_test_1.shape[0], 1, x_test_1.shape[1])
x_test_2 = x_test_2.reshape(x_test_2.shape[0], 1, x_test_2.shape[1])
x_test_3 = x_test_3.reshape(x_test_3.shape[0], 1, x_test_3.shape[1])

# # multiclass
# # y_train=np_utils.to_categorical(y_train)
# # y_test=np_utils.to_categorical(y_test)
# # y_test_21=np_utils.to_categorical(y_test_21)

input_1 = Input(shape=(x_train_1.shape[1],x_train_1.shape[2]))
input_2 = Input(shape=(x_train_2.shape[1],x_train_2.shape[2]))
input_3 = Input(shape=(x_train_3.shape[1],x_train_3.shape[2]))

left = LSTM(120, return_sequences=True)(input_1)
left = LSTM(120, return_sequences=True)(left)
left = Dropout(0.05)(left)

middle = LSTM(120, return_sequences=True)(input_2)
middle = LSTM(120, return_sequences=True)(middle)
middle = Dropout(0.05)(middle)

right = LSTM(120,  return_sequences=True)(input_3)
right = LSTM(120, return_sequences=True)(right)
right = Dropout(0.05)(right)

merged = Concatenate(axis=-1)([left,middle,right])

final = GRU(60, return_sequences=False)(merged)
final = Dropout(0.05)(final)

# final = Flatten()(final)
# binary
predictions = Dense(1, activation='sigmoid')(final)

# # multiclass (nsl = 5 and cicids = 7)
# predictions = Dense(5, activation='softmax')(x)


model = Model(inputs=[input_1,input_2,input_3], outputs=predictions)
model.summary()

# # optimizer
adam = Adam(lr=0.001)

# #binary
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

# #multiclass
# # model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit([x_train_1,x_train_2,x_train_3], y_train, validation_data=([x_test_1,x_test_2,x_test_3], y_test), epochs=100, batch_size=32)

# # save the model
# # model.save("model.hdf5")

loss, accuracy = model.evaluate([x_test_1,x_test_2,x_test_3], y_test, batch_size=32)
# loss_21, accuracy_21 = model.evaluate(x_test_21, y_test_21, batch_size=32)

print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# print("\nLoss 21: %.2f, Accuracy 21: %.2f%%" % (loss_21, accuracy_21*100))

y_pred = model.predict([x_test_1,x_test_2,x_test_3])
# y_pred_21 = model.predict_classes(x_test_21)

print("\nAnomaly in Test: ",np.count_nonzero(y_test, axis=0))
print("\nAnomaly in Prediction: ",np.count_nonzero(y_pred, axis=0))

# print("\nAnomaly in Test 21: ",np.count_nonzero(y_test_21, axis=0))
# print("\nAnomaly in Prediction 21: ",np.count_nonzero(y_pred_21, axis=0))