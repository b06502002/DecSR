# import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# this script checks the multiple output DL framework (One can use Kaggle to utilize free gpu resource)
    # other useful reference: 
    # https://keras.io/guides/training_with_built_in_methods/

image_input = keras.Input(shape=(15, 15, 1), name="img_input")

# x = layers.Rescaling(1.0 / 255)(image_input)
x = layers.Conv2D(32, 3, activation='relu')(image_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(1)(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.GlobalMaxPooling2D()(x)
x = layers.Dense(2048)(x)

offset1_output = layers.Dense(1, name="offset1_output")(x)
offset2_output = layers.Dense(1, name="offset2_output")(x)
FWHM_output    = layers.Dense(1, name="FWHM_output")(x)
ellipx_output  = layers.Dense(1, name="ellipx_output")(x)
ellipy_output  = layers.Dense(1, name="ellipy_output")(x)
skewx_output   = layers.Dense(1, name="skewx_output")(x)
skewy_output   = layers.Dense(1, name="skewy_output")(x)
triax_output   = layers.Dense(1, name="triax_output")(x)
triay_output   = layers.Dense(1, name="triay_output")(x)
kurto_output   = layers.Dense(1, name="kurto_output")(x)

model = keras.Model(
    inputs=image_input, outputs=[offset1_output, offset2_output, FWHM_output, ellipx_output, ellipy_output, skewx_output, skewy_output, triax_output, triay_output, kurto_output]
)
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

