import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Loading the dataset and perform splitting
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Peforming reshaping operation
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    conv1 = Conv2D(6, kernel_size=(5, 5), activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, kernel_size=(5, 5), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flatten = Flatten()(pool2)
    dense1 = Dense(120, activation='relu')(flatten)
    dense2 = Dense(84, activation='relu')(dense1)
    outputs = Dense(10, activation='softmax')(dense2)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128 * strategy.num_replicas_in_sync, epochs=20, verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
model.save('LeNet5GPU.h5')
print(score)
