import numpy as np
import pandas as pd
from keras import datasets
from keras.models import load_model
from keras.src.applications import vgg16, resnet
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16
from keras.src.datasets import cifar10
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from MyVGG16 import getCifar
from SingleTest import checkImageNet, SingleTest


def testSingleImageNet():
    singleTest = SingleTest('test 1')
    if singleTest.imageNetReady is False:
        return
    # singleTest.setImage("tiny-imagenet-200/val/images/val_0.JPEG")
    # sample [('n02948072', 'candle', 0.10180407)]
    # n02948072 is the class number of candle. Use this to verify if the prediction is correct

    singleTest.calValFileAccuracy(iterationNum=3, model=VGG16(weights='imagenet'),
                                  pre_p_fun=vgg16.preprocess_input,
                                  decode_fun=vgg16.decode_predictions, top=3)

    singleTest.calValFileAccuracy(iterationNum=3, model=ResNet50(weights='imagenet'),
                                  pre_p_fun=resnet.preprocess_input,
                                  decode_fun=resnet.decode_predictions, top=3)


def testExistModel(path):
    myModel = load_model(path)
    (x_train, y_train), (x_test, y_test) = getCifar()
    for i in range(30):
        image = np.expand_dims(x_test[i], axis=0)
        prediction = myModel.predict(image)[0]
        real = y_test[i]
        index_max_pre = np.argmax(prediction)
        index_max_real = np.argmax(real)
        if index_max_pre == index_max_real:
            print("match")
        else:
            print("not match")


testExistModel("vgg16_1.keras")
