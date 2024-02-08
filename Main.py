import pandas as pd
from keras import datasets
from keras.src.applications import vgg16, resnet
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from SingleTest import checkImageNet, SingleTest


def main():
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

# Just a sample
def randomTest():
    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    print(model.summary())
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    image = vgg16.preprocess_input(x_train[0].reshape(
        (1, x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2])))
    pred = model.predict(image)

    # decode_predictions expects a prediction of 1000 classes (i.e. a 2D array of shape (samples, 1000))
    label = vgg16.decode_predictions(pred, top=1)

randomTest()