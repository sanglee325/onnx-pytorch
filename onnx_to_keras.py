import onnx
from onnx2keras import onnx_to_keras

import keras
import numpy as np

# Load ONNX model
onnx_path = "ckpt/resnet18.onnx"
onnx_model = onnx.load(onnx_path)


keras_model = onnx_to_keras(onnx_model, ['input.1'])

#keras_model.summary()

num_classes = 10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

x_test = x_test.reshape(10000, 3, 32, 32)
y_pred = keras_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

acc = np.sum(y_pred == y_test.reshape(-1))/10000 * 100
print("Accuracy: %.2f%%" % (acc))

#print("Accuracy: %.2f%%" % (scores[1]*100))