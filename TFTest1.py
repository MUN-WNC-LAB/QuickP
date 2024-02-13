import os

from MyVGG16 import getCifar, initModelForCifar10

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras


def make_or_restore_model():
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint (existing trained model) available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return initModelForCifar10()


def run_training(epochs=1):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()
        (x_train, y_train), (x_test, y_test) = getCifar()
        callbacks = [
            # This callback saves a SavedModel every epoch
            # We include the current epoch in the folder name.
            keras.callbacks.ModelCheckpoint(
                filepath="./ckpt/ckpt-{epoch}.keras",
                save_freq="epoch",
            )
        ]
        model.fit(
            x=x_train, y=y_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
            validation_split=0.4
        )


# Running the first time creates the model
run_training(2)
