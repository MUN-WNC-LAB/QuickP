import pandas as pd
from keras.src.applications import vgg16, resnet
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16

from SingleTest import checkImageNet, SingleTest


# it seems that only val file has the class info (real class) of each image in the val/images file.
def loadValFile():
    val_annotation = pd.read_csv('tiny-imagenet-200/val/val_annotations.txt', sep='\t',
                                 usecols=[0, 1], names=['imagePath', 'label'])
    print(val_annotation.columns)
    print(val_annotation.shape)


def main():
    singleTest = SingleTest('test 1')
    if singleTest.imageNetReady is False:
        return
    singleTest.setImage("tiny-imagenet-200/val/images/val_0.JPEG")
    # sample [('n02948072', 'candle', 0.10180407)]
    # n02948072 is the class number of candle. Use this to verify if the prediction is correct
    singleTest.test_single_of(model=VGG16(weights='imagenet'), pre_p_fun=vgg16.preprocess_input,
                              decode_fun=vgg16.decode_predictions, top=1)
    singleTest.test_single_of(model=ResNet50(weights='imagenet'), pre_p_fun=resnet.preprocess_input,
                              decode_fun=resnet.decode_predictions, top=1)

    loadValFile()


main()
