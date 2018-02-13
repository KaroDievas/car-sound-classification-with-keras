from keras.preprocessing.image import ImageDataGenerator
from Models.KerasModel import KerasModel
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K

K.set_image_dim_ordering('th')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_width, img_height = 256, 256
train_data_dir = 'Data/big_pictures/train'
validation_data_dir = 'Data/big_pictures/validation'
nb_train_samples = 50
nb_validation_samples = 50
nb_epoch = 50

kerasModel = KerasModel()
model = kerasModel.get_model(img_width, img_height)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=5,
    class_mode='categorical')

class_dictionary = train_generator.class_indices

print('class dictionary', class_dictionary)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=5,
    class_mode='categorical')

check_pointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    callbacks=[check_pointer])
