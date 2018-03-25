from keras.preprocessing.image import ImageDataGenerator
from Models.KerasModel import KerasModel
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

K.set_image_dim_ordering('th')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_width, img_height = 503, 376
train_data_dir = 'Data/train'
validation_data_dir = 'Data/validation'
nb_train_samples = 3700
nb_validation_samples = 3700
nb_epoch = 30
batch_size = 30

kerasModel = KerasModel()
model = kerasModel.get_model(img_width, img_height)

model.compile(loss='binary_crossentropy',
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
    classes=['audi', 'other'],
    batch_size=batch_size,
    class_mode='binary')

class_dictionary = train_generator.class_indices

print('class dictionary', class_dictionary)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    classes=['audi', 'other'],
    batch_size=batch_size,
    class_mode='binary')

check_pointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples/batch_size,
    callbacks=[check_pointer])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
