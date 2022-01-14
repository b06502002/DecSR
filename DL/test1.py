import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# https://keras.io/guides/functional_api/

# this script checks the multiple output DL framework
    # other useful reference: 
    # https://stackoverflow.com/questions/50525747/keras-show-loss-for-each-label-in-a-multi-label-regression
    # https://keras.io/guides/training_with_built_in_methods/

inputs = keras.Input(shape=(784,))
# img_inputs = keras.Input(shape=(32, 32, 3))# Just for demonstration purposes.
print(inputs.shape, inputs.dtype)

dense = layers.Dense(64, activation="relu")
x = dense(inputs)# create a new node in the graph of layers by calling a layer on this inputs object
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
print(model.summary())
keras.utils.plot_model(model, "my_first_model.png")
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

#=== Train, evaluate and infer ===

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
#===

