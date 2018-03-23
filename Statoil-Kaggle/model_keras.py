import json
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


def function_preprocessing(x, y):
    x = np.array(x).reshape(75, 75)
    y = np.array(y).reshape(75, 75)
    z = np.array(x) + np.array(y)

    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    z = (z - z.min()) / (z.max() - z.min())

    result = np.dstack((x, y, z))
    return result.reshape(75*75*3)

class ModelCnn:

    def __init__(self, data_path='/home/vaden4d/Documents/kaggles/iceberg'):
        file = open(data_path + '/train.json')
        self.data = json.load(file)
        self.images = []
        self.labels = []

        self.probability = 0.2

    def data_formatting(self, func=function_preprocessing, num_classes=2):
        for train_sample in self.data:
            self.images.append(func(train_sample['band_1'], train_sample['band_2']))
            self.labels.append(np.eye(num_classes)[train_sample['is_iceberg']])

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.data = self.images, self.labels

    def augmentation(self):
        augmented_images = []
        augmented_labels = []
        for image, label in zip(self.images, self.labels):

            image_0 = image[:, :, 0]
            image_1 = image[:, :, 1]
            image_2 = image[:, :, 2]

            image_0_90 = np.rot90(image_0, 1)
            image_0_180 = np.rot90(image_0, 2)
            image_0_270 = np.rot90(image_0, 3)

            image_1_90 = np.rot90(image_1, 1)
            image_1_180 = np.rot90(image_1, 2)
            image_1_270 = np.rot90(image_1, 3)

            image_2_90 = np.rot90(image_2, 1)
            image_2_180 = np.rot90(image_2, 2)
            image_2_270 = np.rot90(image_2, 3)

            augmented_images.append(np.dstack((image_0_90, image_1_90, image_2_90)))
            augmented_labels.append(label)

            augmented_images.append(np.dstack((image_0_180, image_1_180, image_2_180)))
            augmented_labels.append(label)

            augmented_images.append(np.dstack((image_0_270, image_1_270, image_2_270)))
            augmented_labels.append(label)

        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        augmented_images = augmented_images.reshape((augmented_images.shape[0], 75, 75, 3))

        self.images = np.append(self.images, augmented_images, axis=0)
        self.labels = np.append(self.labels, augmented_labels, axis=0)

    def train_boosting(self):
        self.model = XGBClassifier()
        self.model.fit(self.train_data_x, self.train_data_l[:, 1])

    def build_model(self):

        self.model = Sequential()
        # CNN 1
        self.model.add(Conv2D(16, kernel_size=(5, 5), activation='elu', input_shape=(75, 75, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        # CNN 2
        self.model.add(Conv2D(32, kernel_size=(4, 4), activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        # CNN 3
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        # CNN 4
        self.model.add(Conv2D(128, kernel_size=(2, 2), activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        # CNN 5
        self.model.add(Conv2D(256, kernel_size=(2, 2), activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        # Dense 1
        self.model.add(Dense(512, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 1
        self.model.add(Dense(400, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 2
        self.model.add(Dense(256, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 3
        self.model.add(Dense(128, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 4
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 5
        self.model.add(Dense(32, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 6
        self.model.add(Dense(16, activation='elu'))
        self.model.add(Dropout(0.2))
        # Dense 7
        self.model.add(Dense(8, activation='elu'))
        self.model.add(Dropout(0.2))
        # Output
        self.model.add(Dense(2, activation="softmax"))
        optimizer = Adam(lr=0.001, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train_test_split(self):
        self.train_data_x, self.test_data_x, self.train_data_l, self.test_data_l = train_test_split(self.images,
                                                                                                    self.labels,
                                                                                                    test_size=0.3
                                                                                                    )
    def train(self):

        self.model.summary()
        batch_size = 20
        epochs = 100

        earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('second.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4,
                                           mode='min')
        
	self.model.fit(self.images, self.labels, batch_size=batch_size, epochs=epochs, verbose=1,
                  callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)

    def load_best(self):

        self.model.load_weights(filepath='second.mdl_wts.hdf5')

        score = self.model.evaluate(self.images, self.labels, verbose=1)
        print('Train score:', score[0])
        print('Train accuracy:', score[1])

    def scores(self, data_path='/home/vaden4d/Documents/kaggles/iceberg', func=function_preprocessing):
        file = open(data_path + '/test.json')
        test_data = json.load(file)

        file_test = open('submission_new', 'w+')
        file_test.write('id,is_iceberg' + '\n')

        for sample in test_data:
            img = func(sample['band_1'], sample['band_2']).reshape(1, 75, 75, 3)
            result = self.model.predict(img)[0][1]
            s = str(sample['id']) + ',' + str(result) + '\n'
            print(s)
            file_test.write(s)


obj = ModelCnn()
obj.data_formatting()
#obj.augmentation()
obj.train_test_split()
#obj.build_model()
obj.train()
obj.load_best()
obj.scores()
