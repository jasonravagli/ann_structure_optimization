import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold


class AnnKFold:
    def __init__(self, n_fold):
        self.model = None
        self.x_train_set = None
        self.y_train_set = None
        self.x_test_set = None
        self.y_test_set = None
        self.n_fold = n_fold

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

        # if n_classes > 2:
        #     self.model.add(Dense(n_classes, activation='softmax'))
        #     self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # else:
        #     self.model.add(Dense(n_classes, activation='sigmoid'))
        #     self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        learning_rate = 0.00158  # Learning rate from article
        if n_classes > 2:
            self.model.add(Dense(n_classes, activation='softmax'))
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model.add(Dense(n_classes, activation='sigmoid'))
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        """
        Train the network using the k-fold cross validation strategy
        :return: The average accuracy on validation sets selected by k-fold
        """
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20, min_delta=0.005, verbose=1)

        # Strtatified k-fold
        validation_accuracy = 0
        skf = StratifiedKFold(n_splits=self.n_fold)
        for train_indexes, validation_indexes in skf.split(self.x_train_set, self.y_train_set):
            x_train, y_train = self.x_train_set[train_indexes], keras.utils.to_categorical(self.y_train_set[train_indexes])
            x_validation, y_validation = self.x_train_set[validation_indexes], keras.utils.to_categorical(self.y_train_set[validation_indexes])

            history = self.model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                                     epochs=500, batch_size=32, verbose=0, callbacks=[es])

            validation_accuracy = validation_accuracy + history.history['val_accuracy'][-1]

        return validation_accuracy / self.n_fold

    def evaluate_model(self):
        index_accuracy = self.model.metrics_names.index('accuracy')
        return self.model.evaluate(self.x_test_set, self.y_test_set)[index_accuracy]
