from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


class Ann:

    def __init__(self):
        self.model = None
        self.x_train_set = None
        self.y_train_set = None
        self.x_valid_set = None
        self.y_valid_set = None
        self.x_test_set = None
        self.y_test_set = None

    def create_model(self, n_layers, n_neurons, input_dimension, n_classes):
        """
        Create an ANN for classification plroblems having n_layers hidden layers (dense layers) and n_neurons neurons per layer
        :param n_layers: number of hidden dense layers
        :param n_neurons: number of neurons per layer
        :param input_dimension: integer specifying the input dimension
        :param n_classes: number of feature classes (equals to the number of neurons the output layer)
        :return:
        """

        assert n_classes >= 2

        self.model = Sequential()

        for i in range(n_layers):
            if i == 0:
                self.model.add(Dense(n_neurons, input_dim=input_dimension, activation='relu'))
            else:
                self.model.add(Dense(n_neurons, activation='relu'))

        if n_classes > 2:
            self.model.add(Dense(n_classes, activation='softmax'))
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.add(Dense(n_classes, activation='sigmoid'))
            self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=50, min_delta=0.005, verbose=1)
        self.model.fit(self.x_train_set, self.y_train_set, validation_data=(self.x_valid_set, self.y_valid_set),
                       epochs=500, batch_size=32, verbose=0, callbacks=[es])

    def validate_model(self):
        # index_loss = self.model.metrics_names.index('loss')
        # return self.model.evaluate(self.x_valid_set, self.y_valid_set)[index_loss]
        index_accuracy = self.model.metrics_names.index('accuracy')
        return self.model.evaluate(self.x_valid_set, self.y_valid_set)[index_accuracy]

    def evaluate_model(self):
        index_accuracy = self.model.metrics_names.index('accuracy')
        return self.model.evaluate(self.x_test_set, self.y_test_set)[index_accuracy]
