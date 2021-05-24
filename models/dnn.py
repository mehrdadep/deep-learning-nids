from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential


class DNNModel:

    @classmethod
    def model(cls, run_type, shapes):
        model = Sequential()
        model.add(Dense(128, input_shape=shapes))
        model.add(Dropout(0.1))
        model.add(Dense(256))
        model.add(Dropout(0.1))
        model.add(Dense(128))
        model.add(Dropout(0.1))
        model.add(Flatten())
        #
        if run_type == 0:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
        else:
            model.add(Dense(5))
            model.add(Activation('softmax'))

        return model
