from keras.src.applications import vgg16, resnet
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16

from Test import checkImageNet, Test


def main():
    test = Test('test 1')
    if test.imageNetReady is False:
        return

    test.test_single_of(model=VGG16(weights='imagenet'), pre_p_fun=vgg16.preprocess_input,
                        decode_fun=vgg16.decode_predictions)
    test.test_single_of(model=ResNet50(weights='imagenet'), pre_p_fun=resnet.preprocess_input,
                        decode_fun=resnet.decode_predictions)


main()
