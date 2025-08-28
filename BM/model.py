import tensorflow as tf
from tensorflow.keras import layers, models, activations

class Net(tf.keras.Model):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = layers.Dense(200, input_shape=(input_size,), activation='relu')
        self.fc2 = layers.Dense(200, activation='relu')
        self.fc3 = layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        x = self.fc1(x)
        sum1 = tf.reduce_mean(x, axis=0)
        x = self.fc2(x)
        sum2 = tf.reduce_mean(x, axis=0)
        x = self.fc3(x)
        sum3 = tf.reduce_mean(x, axis=0)
        return x, sum1, sum2, sum3

    def mask_forward(self, x, index1=None, index2=None):
        x = self.fc1(x)
        if index1 is not None:
            mask = tf.ones_like(x)
            mask = tf.tensor_scatter_nd_update(mask, 
                                               indices=tf.expand_dims(index1, axis=1), 
                                               updates=tf.zeros(len(index1)))
            x = x * mask
        x = self.fc2(x)
        if index2 is not None:
            mask = tf.ones_like(x)
            mask = tf.tensor_scatter_nd_update(mask, 
                                               indices=tf.expand_dims(index2, axis=1), 
                                               updates=tf.zeros(len(index2)))
            x = x * mask
        x = self.fc3(x)
        return x

    def neuron_sum(self, x):
        x = self.fc1(x)
        sum1 = tf.reduce_mean(x, axis=0)
        x = self.fc2(x)
        sum2 = tf.reduce_mean(x, axis=0)
        x = self.fc3(x)
        sum3 = tf.reduce_mean(x, axis=0)
        return sum1, sum2, sum3