import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
# ref: https://keras.io/examples/keras_recipes/tfrecord/

class TrainingScript:
    def __init__(self):
        self.init_lr = 0.00005
        self.DI = { 0:"AnnualCrop", 1:"Forest", 2:"HerbaceousVegetation", 3:"Highway", 4:"Industrial", 5:"Pasture", 6:"PermanentCrop", 7:"Residential", 8:"River", 9:"SeaLake"}
        self.BATCH_SIZE = 16
        self.IMAGE_SIZE = [64, 64]
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def get_dataset(self, filenames, val_split=None):
        def read_tfrecord(example):
            tfrecord_format = (
                {
                    "raw_image": tf.io.FixedLenFeature([], tf.string),
                    "label": tf.io.FixedLenFeature([], tf.int64),
                }
            )
            example = tf.io.parse_single_example(example, tfrecord_format)

            self.image = tf.io.decode_raw(example['raw_image'], out_type=tf.uint8)            
            self.label = example['label']
            return tf.reverse(tf.reshape(self.image, [64, 64, 3]), axis=[-1]), tf.cast(self.label, tf.int32)

        def load_dataset(filenames, split=1):
            ignore_order = tf.data.Options()
            ignore_order.experimental_deterministic = False  # disable order, increase speed
            dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
            dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
            dataset_train = dataset.take(int(split*18900))
            dataset_val = dataset.skip(int(split*18900))
            dataset_train, dataset_val = dataset_train.map(partial(read_tfrecord), num_parallel_calls=self.AUTOTUNE), dataset_val.map(partial(read_tfrecord), num_parallel_calls=self.AUTOTUNE)
            return dataset_train, dataset_val
    
        dataset1, dataset2 = load_dataset(filenames, val_split)
        dataset1, dataset2 = dataset1.shuffle(512), dataset2.shuffle(512)
        dataset1, dataset2 = dataset1.prefetch(buffer_size=self.AUTOTUNE), dataset2.prefetch(buffer_size=self.AUTOTUNE)
        dataset1, dataset2 = dataset1.batch(self.BATCH_SIZE), dataset2.batch(self.BATCH_SIZE)
        return dataset1, dataset2

    def show_batch(self, image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n] / 255.0)
            plt.title(self.DI.get(label_batch[n]))
            plt.axis("off")
        plt.show()
    
    def make_model(self):
        image_input = keras.Input(shape=(15, 15, 1), name="img_input")
        x = layers.Conv2D(32, 3, padding='same',activation='relu')(image_input)
        x = layers.MaxPooling2D(2)(x)
        print(x.shape)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        print(x.shape)
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.GlobalMaxPooling2D()(x)
        print(x.shape)
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

        self.model = keras.Model(
            inputs=image_input, outputs=[offset1_output, offset2_output, FWHM_output, ellipx_output, ellipy_output, skewx_output, skewy_output, triax_output, triay_output, kurto_output]
        )
        
        initial_learning_rate = self.init_lr
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=[keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"),
                        keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
                    ])
        return self.model

def main(lr):
    # DATA_PATH = "./data_tfrec"

    # Train_NAME = tf.io.gfile.glob(DATA_PATH + "/SAT_train.tfrecord")
    # train_dataset, valid_dataset = TrainingScript().get_dataset(Train_NAME, 0.7)
    # TrainingScript().init_lr = lr

    checkpoint_cb = keras.callbacks.ModelCheckpoint("./models/m1.h5", save_best_only=False)
    logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S-")

    model = TrainingScript().make_model()
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    history = model.fit(
        train_dataset,
        epochs=3000,
        validation_data=valid_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    )

    # image_batch, label_batch = next(iter(train_dataset))
    # TrainingScript().show_batch(image_batch, label_batch)
    return 0

if __name__ == "__main__":
    if os.environ['CONDA_DEFAULT_ENV'] == "tf-proj":
        Lr = input("initial learning rate: ")
        main(Lr)
    else:
        print("Wrong environment")
