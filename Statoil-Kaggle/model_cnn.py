import tensorflow as tf
import json
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os

#normalizing and reshaping channels
#x, y - channels that consist of images with horizontal and vertical polarization
def function_preprocessing(x, y):
    x = np.array(x).reshape(75, 75)
    y = np.array(y).reshape(75, 75)
    z = np.array(x) + np.array(y)

    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    z = (z - z.min()) / (z.max() - z.min())

    result = np.dstack((x, y, z))
    return result

#Class consists of model and training on the TensorFlow for the Kaggle task - Statoil Iceberg classification
#Training saves every epoch in separate folder (name of the folder locates in the training method)
class ModelConv:
    #init method with some constants
    def __init__(self, data_path='/home/vaden4d/Documents/kaggles/iceberg'):
        file = open(data_path + '/train.json')
        self.data = json.load(file)
        self.images = []
        self.labels = []

        self.x = tf.placeholder(tf.float32, [None, 75, 75, 3], name='x')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.y = tf.placeholder(tf.float32, [None, 2], name='y')
        self.training = tf.placeholder(tf.bool, name='training')

        self.probability = 0.2

    #download neccecary data
    def data_formatting(self, func=function_preprocessing, num_classes=2):

        for train_sample in self.data:
            self.images.append(func(train_sample['band_1'], train_sample['band_2']))
            self.labels.append(np.eye(num_classes)[train_sample['is_iceberg']])

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.data = self.images, self.labels
    
    #flips and rotation augmentation for better generalization
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

    def batch_generator(self, batch_size):
        pass

    def train_test_split(self):
        self.train_data_x, self.test_data_x, self.train_data_l, self.test_data_l = train_test_split(self.images,
                                                                                                    self.labels,
                                                                                                    test_size=0.3
                                                                                                    )
    #creating model architecture
    #experimenting with new TF calls of keras (new feature of TF 1.4) 
    def evaluate_model(self, input_x):
        initializer = tf.keras.initializers.RandomNormal(seed=41)
        #Block 1
        input_x = tf.keras.layers.Conv2D(16,
                                         kernel_size=(5, 5),
                                         strides=(1, 1),
                                         padding='valid',
                                         use_bias=False,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer)(input_x)
        input_x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
	      #strange behavior of batchnorm - omg - correct or check it later
        #input_x = tf.contrib.layers.batch_norm(input_x, is_training=self.training)
        input_x = tf.keras.layers.Activation('relu')(input_x)
        #Block 2
        input_x = tf.keras.layers.Conv2D(32,
                                         kernel_size=(4, 4),
                                         strides=(1, 1),
                                         padding='valid',
                                         use_bias=False,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer
                                         )(input_x)

        input_x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('relu')(input_x)
        #Block 3
        input_x = tf.keras.layers.Conv2D(64,
                                         kernel_size=(3, 3),
                                         strides=(1, 1),
                                         padding='valid',
                                         use_bias=True,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer)(input_x)
        input_x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('relu')(input_x)
        #Block 4
        input_x = tf.keras.layers.Conv2D(128,
                                         kernel_size=(2, 2),
                                         strides=(1, 1),
                                         padding='valid',
                                         use_bias=True,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer)(input_x)
        input_x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        #input_x = tf.contrib.layers.batch_norm(input_x, is_training=self.training)
        input_x = tf.keras.layers.Activation('relu')(input_x)
        #Block 5
        input_x = tf.keras.layers.Conv2D(256,
                                         kernel_size=(2, 2),
                                         strides=(1, 1),
                                         padding='valid',
                                         use_bias=True,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer)(input_x)
        input_x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('relu')(input_x)
        
        #automatical flattenization
        input_x = tf.reshape(input_x, shape=[-1, reduce(lambda x, y: x * y, input_x.shape[1:])])
        #Dense Block 1
        input_x = tf.keras.layers.Dense(256,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer)(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('elu')(input_x)  
        
        #Dense Block 2
        input_x = tf.keras.layers.Dense(128,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer)(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('elu')(input_x)
        #Dense Block 3
        input_x = tf.keras.layers.Dense(64,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer)(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('elu')(input_x)
        #Dense Block 4
        input_x = tf.keras.layers.Dense(32,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer)(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('elu')(input_x)
        #Dense Block 5
        input_x = tf.keras.layers.Dense(16,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer)(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('elu')(input_x)

        #Finall
        input_x = tf.keras.layers.Dense(2,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer)(input_x)
        input_x = tf.layers.dropout(input_x, self.probability, training=self.training)
        input_x = tf.keras.layers.Activation('softmax')(input_x)

        return input_x

    def build_model(self):
        
        self.output_layer = self.evaluate_model(self.x)
        self.loss = -tf.reduce_mean(tf.reduce_sum(tf.log(self.output_layer) * self.y, axis=1))

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer').minimize(self.loss)

        self.prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1), name='prediction')
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, "float"), name='accuracy')

    def train(self):

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

        self.error_train = []
        self.error_test = []

        self.error_train_all = []
        self.error_test_all = []

        epoch = 4
        batch_size = 40

        with tf.Session() as sess:
            sess.run(init)

            for i in range(epoch):

                self.shuffle_train_test()

                for j in range(int(len(self.train_data_x) / batch_size)):

                    batch_x, batch_y = self.train_data_x[j:j + batch_size], self.train_data_l[j:j + batch_size]
                    sess.run(self.optimizer, feed_dict={
                                                        self.learning_rate: 0.1*0.5**(i//30),
                                                        self.x: batch_x,
                                                        self.y: batch_y,
                                                        self.training: True
                    })

                    c_train = sess.run(self.loss, feed_dict={
                                                        self.x: batch_x,
                                                        self.y: batch_y,
                                                        self.training: False})
                    self.error_train.append(c_train)
                    self.accuracy_train = self.accuracy.eval({
                                                        self.x: batch_x,
                                                        self.y: batch_y,
                                                        self.training: False})


                    batch_x, batch_y = self.test_data_x[:batch_size], self.test_data_l[:batch_size]
                    c_test = sess.run(self.loss, feed_dict={
                                                        self.x: batch_x,
                                                        self.y: batch_y,
                                                        self.training: False})
                    self.error_test.append(c_test)
                    self.accuracy_test = self.accuracy.eval({
                                                        self.x: batch_x,
                                                        self.y: batch_y,
                                                        self.training: False})
                    print('---', i + 1, 'epoch ---', j, 'batch')
                    print('---', j, 'train:', self.accuracy_train)
                    print('---', j, 'test: ', self.accuracy_test)
                    print('--- ---Loss-train:', c_train)
                    print('--- ---Loss--test:', c_test)


                train_c = 0
                for k in range(len(self.train_data_x)//batch_size):
                    batch_x, batch_y = self.train_data_x[k:k+batch_size], self.train_data_l[k:k+batch_size]
                    train_c += sess.run(self.loss, feed_dict={
                                                            self.x: batch_x,
                                                            self.y: batch_y,
                                                            self.training: False})

                train_c = batch_size * train_c / len(self.train_data_x)
                self.error_train_all.append(train_c)
                print('TRAIN LOSS:', train_c)

                test_c = 0
                for k in range(len(self.test_data_x) // batch_size):
                    batch_x, batch_y = self.test_data_x[k:k + batch_size], self.test_data_l[k:k + batch_size]
                    test_c += sess.run(self.loss, feed_dict={
                                                            self.x: batch_x,
                                                            self.y: batch_y,
                                                            self.training: False})
                test_c = batch_size * test_c / len(self.test_data_x)
                self.error_test_all.append(test_c)
                print('TEST LOSS:', test_c)

                #saving model every epoch
                path = 'models-29-11-1/epoch-'+str(i)+'-train-'+str(train_c)+'-test-'+str(test_c)
                os.makedirs(path)
                saver.save(sess, os.path.join(os.getcwd(),
                                              path+'/epoch-'+str(i)+'-train-'+str(train_c)+'-test-'+str(test_c)))

    def plot(self):
        plt.plot(self.error_train, label='train')
        plt.plot(self.error_test, label='test')
        plt.legend()
        plt.show()
        plt.plot(self.error_test_all, label='test all')
        plt.plot(self.error_train_all, label='train all')
        plt.legend()
        plt.show()


obj = ModelConv()
obj.data_formatting()
obj.augmentation()
obj.build_model()
obj.train_test_split()
obj.train()
obj.plot()
