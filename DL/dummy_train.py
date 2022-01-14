import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

image_input = keras.Input(shape=(15, 15, 1), name="img_input")

x = layers.Conv2D(32, 3, padding='same',activation='relu')(image_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(2048)(x)

offset1_output = layers.Dense(1, name="offset1_output")(x)
offset2_output = layers.Dense(1, name="offset2_output")(x)


initial_learning_rate = 0.03
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)


model = keras.Model(inputs=image_input, outputs=[offset1_output, offset2_output])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=[
                    keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                    keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")])


maxGrayscale = 255
# Dummy input data
dummy_img_data = np.random.randint(maxGrayscale, size=(320,15,15))

# Dummy target data
offset1 = np.random.random(size=(320, 1))
offset2 = np.random.random(size=(320, 1))

checkpoint_cb = keras.callbacks.ModelCheckpoint("./models/m1.h5", save_best_only=False)
logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S-")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
model.fit(
    {"img_input": dummy_img_data},
    {"offset1_output": offset1, "offset2_output": offset2},
    epochs=1000,
    batch_size=32,
    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback]
)