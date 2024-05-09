import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

# Load the saved model
model = load_model('LeNet5.h5')

# Create symbolic tensors for input and output
input_tensor = model.inputs[0]
output_tensor = model.outputs[0]

# Get tensor names
input_tensor_name = input_tensor.name.split(':')[0]
output_tensor_name = output_tensor.name.split(':')[0]

print("Input tensor name:", input_tensor_name)
print("Output tensor name:", output_tensor_name)

# Loading the dataset and perform splitting
(_, _), (x_test, y_test) = mnist.load_data()
# Peforming reshaping operation
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255  # Normalize pixel values to [0, 1]

# Convert class vectors to binary class matrices
y_test = to_categorical(y_test, num_classes=10)

predictions = model.predict(x_test)

# Example: Print the first 10 predictions
for i in range(10):
    print("Predicted:", np.argmax(predictions[i]))
    print("Actual:", np.argmax(y_test[i]))