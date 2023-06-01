import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from model_trainer import generateModel

IMG_SIZE = 16


def saveModel(model):
    # save model
    model.save("mnist_test")


if __name__ == '__main__':

    print('generate a new model?\ny/n')
    if input() == 'y':

        # generate model
        model, test_images, test_labels = generateModel()

        print('save the new model?\nit will overrite the older one\n y/n')
        if input() == 'y':
            saveModel()

        # load model from file-system
        model = tf.keras.models.load_model('mnist_test')
