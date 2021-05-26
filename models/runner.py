import time

import numpy as np
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix

from models.conv1d import Conv1DModel
from models.dnn import DNNModel
from models.gru import GRUModel
from models.lstm import LSTMModel
from models.rnn import RNNModel
from services.process import Processor


class Runner:

    @classmethod
    def run(cls, run_type, dataset, model_type, epochs):
        x_train, y_train, x_test, y_test = Processor.get_data(
            run_type,
            dataset,
        )

        # reshape input to be [samples, timesteps, features]
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        if run_type == 1:
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)

        start = time.time()
        if model_type == 1:
            model = Conv1DModel.model(
                run_type,
                dataset,
                (x_train.shape[1], x_train.shape[2]),
            )
        elif model_type == 2:
            model = DNNModel.model(
                run_type,
                dataset,
                (x_train.shape[1], x_train.shape[2]),
            )
        elif model_type == 3:
            model = GRUModel.model(
                run_type,
                dataset,
                (x_train.shape[1], x_train.shape[2]),
            )
        elif model_type == 4:
            model = LSTMModel.model(
                run_type,
                dataset,
                (x_train.shape[1], x_train.shape[2]),
            )
        elif model_type == 5:
            model = RNNModel.model(
                run_type,
                dataset,
                (x_train.shape[1], x_train.shape[2]),
            )

        model.summary()

        # optimizer
        adam = Adam(lr=0.0005)

        if run_type == 0:
            model.compile(
                optimizer=adam,
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
        else:
            model.compile(
                optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )

        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=32,
        )

        # save the model
        # model.save(f"model{run_type}.hdf5")

        print("--- %s seconds ---" % (time.time() - start))

        loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
        print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

        y_pred = model.predict_classes(x_test)
        print("\nAnomaly in Test: ", np.count_nonzero(y_test, axis=0))
        print("\nAnomaly in Prediction: ", np.count_nonzero(y_pred, axis=0))

        if run_type == 0:
            print('Confusion Matrix')
            print(confusion_matrix(y_test, y_pred))
            print('Classification Report')
            print(classification_report(y_test, y_pred))
