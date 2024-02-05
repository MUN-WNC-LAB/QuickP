from zipfile import ZipFile

import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def checkCatDog():
    path_cat_dog = 'kagglecatsanddogs_5340'
    path_cat_dog_zip = 'kagglecatsanddogs_5340.zip'

    if os.path.exists(path_cat_dog):
        print("the imagenet is here")
        return True

    if os.path.exists(path_cat_dog_zip):
        print("need to extract the file")
        with ZipFile(path_cat_dog_zip, 'r') as zip_ref:
            zip_ref.extractall()
        return True

    # file is not here, download that
    print("need to download the file")
    return False


def getCatDogDF():
    if not checkCatDog():
        return
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = "kagglecatsanddogs_5340/PetImages/" + folder_name
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")

    image_size = (180, 180)
    batch_size = 128

    # generating training and test set
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "kagglecatsanddogs_5340/PetImages",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

def initModel():
    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    # fully connected layer
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


class MyVGG16:
    def __init__(self, name):
        self.name = name
        self.model = initModel()
        self.model.summary()

    def getCatDogData(self):
        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(directory="data", target_size=(224, 224))
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(directory="test", target_size=(224, 224))

    def train(self):
        checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
        # hist = self.model.fit(steps_per_epoch=100, generator=traindata, validation_data=testdata,
        #                      validation_steps=10, epochs=100, callbacks=[checkpoint, early])

getCatDogDF()
#myVGG16 = MyVGG16("")
