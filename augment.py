import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
""" tf.keras.layers.preprocessing.RandomHeight

# Create preprocessing layers
height = tf.keras.layers.preprocessing.RandomHeight(0.3)
width = tf.keras.layers.preprocessing.RandomWidth(0.3)
zoom = tf.keras.layers.preprocessing.RandomZoom(0.3)
flip = tf.keras.layers.preprocessing.RandomFlip("horizontal_and_vertical")
rotate = tf.keras.layers.preprocessing.RandomRotation(0.2)
translation = tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)

trainAug = tf.keras.Sequential([
	height, width, zoom, flip, rotate, translation
])

def augment(image, label):
    return trainAug(image), label """

def augment():
    datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

    # featurewise_center=False,  # set input mean to 0 over the dataset
    # samplewise_center=False,  # set each sample mean to 0
    # featurewise_std_normalization=False,  # divide inputs by std of the dataset
    # samplewise_std_normalization=False,  # divide each input by its std
    # zca_whitening=False,  # apply ZCA whitening
    # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # width_shift_range=0.1,  # randomly shift images horizontally
    # height_shift_range=0.1,  # randomly shift images vertically
    # horizontal_flip=True,  # randomly flip images
    # vertical_flip=False)  # randomly flip images

    return datagen

