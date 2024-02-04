import numpy as np
import pandas as pd
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import preprocess_input, decode_predictions
import os, sys, wget
from zipfile import ZipFile

from keras.src.utils import load_img, img_to_array
from matplotlib import pyplot as plt
from keras.datasets import mnist, cifar10
from keras.applications import VGG16


def getValFile():
    val_annotation = pd.read_csv('tiny-imagenet-200/val/val_annotations.txt', sep='\t',
                                 usecols=[0, 1], names=['imagePath', 'label'])
    return val_annotation


def getImageNet():
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    wget.download(url, out=os.getcwd())


def checkImageNet():
    path_imageNet = 'tiny-imagenet-200'
    path_imageNetZip = 'tiny-imagenet-200.zip'

    if os.path.exists(path_imageNet):
        print("the imagenet is here")
        return True

    if os.path.exists(path_imageNetZip):
        print("need to extract the file")
        with ZipFile(path_imageNetZip, 'r') as zip_ref:
            zip_ref.extractall()
        return True

    # file is not here, download that
    print("need to download the file")
    getImageNet()
    return False


class SingleTest:
    def __init__(self, name):
        self.name = name
        self.imageNetReady = checkImageNet()
        # default shape=(224, 224, 3)
        self.singleTestImage = img_to_array(
            load_img('tiny-imagenet-200/test/images/test_0.JPEG', target_size=(224, 224)))
        self.valueFileData = getValFile()

    # only use image size of (224, 224, 3). However, there will be a lot of mess
    def test_single_of(self, model, pre_p_fun, decode_fun, top):
        image = pre_p_fun(self.singleTestImage.reshape(
            (1, self.singleTestImage.shape[0], self.singleTestImage.shape[1], self.singleTestImage.shape[2])))
        pred = model.predict(image)
        # decode_predictions expects a prediction of 1000 classes (i.e. a 2D array of shape (samples, 1000))
        label = decode_fun(pred, top=top)
        print(label)

    def setImage(self, path):
        if not os.path.exists(path):
            print("wrong path")
            return
        self.singleTestImage = img_to_array(
            load_img(path, target_size=(224, 224)))

    def calValFileAccuracy(self, iterationNum, model, pre_p_fun, decode_fun, top):
        # for i in range(valFile.shape[0])
        for i in range(iterationNum):
            filePath = "tiny-imagenet-200/val/images/" + self.valueFileData['imagePath'][i]
            self.setImage(filePath)
            self.test_single_of(model=model, pre_p_fun=pre_p_fun,
                                decode_fun=decode_fun, top=top)
            print("The real class is " + self.valueFileData['label'][i])
