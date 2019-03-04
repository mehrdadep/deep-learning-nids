from DataProcess import DataProcess
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, SimpleRNN, Input, Concatenate, GRU, Conv1D,Flatten
import keras 
import numpy as np
from keras.utils import np_utils
import time
from sklearn.metrics import classification_report, confusion_matrix
import winsound

# get and process data
data = DataProcess()
# x_train, y_train, x_test, y_test = data.return_processed_cicids_data_multiclass()
x_train, y_train, x_test, y_test = data.return_processed_cicids_data_binary()


# reshape input to be [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

# # multiclass
# y_train=np_utils.to_categorical(y_train)
# y_test=np_utils.to_categorical(y_test)

input_1 = Input(name='left_input', shape=(x_train.shape[1],x_train.shape[2]))
input_2 = Input(name='middle_input',shape=(x_train.shape[1],x_train.shape[2]))
input_3 = Input(name='right_input',shape=(x_train.shape[1],x_train.shape[2]))

left = LSTM(120, return_sequences=True)(input_1)
left = LSTM(120, return_sequences=True)(left)
left = LSTM(120, return_sequences=True)(left)
left = Dropout(0.2)(left)

middle = SimpleRNN(120, return_sequences=True)(input_2)
middle = SimpleRNN(120, return_sequences=True)(middle)
middle = SimpleRNN(120, return_sequences=True)(middle)
middle = Dropout(0.2)(middle)

right = GRU(120, return_sequences=True)(input_3)
right = GRU(120, return_sequences=True)(right)
right = GRU(120, return_sequences=True)(right)
right = Dropout(0.2)(right)

merged = Concatenate(axis=-1)([left,middle,right])

final = Conv1D(120, kernel_size=1)(merged)
final = Flatten()(final)

# binary
predictions = Dense(1, name='prediction_layer',activation='sigmoid')(final)

# # multiclass (nsl = 5 and cicids = 7)
# predictions = Dense(5, name='prediction_layer', activation='softmax')(final)


model = Model(inputs=[input_1,input_2,input_3], outputs=predictions)
model.summary()

# # optimizer
adam = Adam(lr=0.0001)

# #binary
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

# #multiclass
# model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model.fit([x_train,x_train,x_train], y_train, validation_data=([x_test,x_test,x_test], y_test), epochs=30, batch_size=32)

# # save the model
# # model.save("model.hdf5")

loss, accuracy = model.evaluate([x_test,x_test,x_test], y_test, batch_size=32)

print('Loss',loss,"Accuaracy",accuracy)

print("--- %s seconds ---" % (time.time() - start))


y_pred = model.predict([x_test,x_test,x_test])
y_pred = [np.round(x) for x in y_pred]
y_pred = np.array(y_pred)
print('Confusion Matrix')
#binary
print(confusion_matrix(y_test, y_pred))
#multiclass
# print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print('Classification Report')
print(classification_report(y_test, y_pred))
winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
