from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, Flatten



class Conv1DModel:

    @classmethod
    def model(cls, run_type, shapes):
        model = Sequential()
        model.add(
            Conv1D(60, kernel_size=1, activation='relu', input_shape=shapes)
        )
        model.add(Dropout(0.1))
        model.add(Conv1D(60, kernel_size=1, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Conv1D(60, kernel_size=1, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Flatten())

        if run_type == 0:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
        else:
            model.add(Dense(5))
            model.add(Activation('softmax'))

        return model
