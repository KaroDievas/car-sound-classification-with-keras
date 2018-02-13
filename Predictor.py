from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import sys
from Models import KerasModel
import tensorflow as tf
from keras import backend as K

K.set_image_dim_ordering('th')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_width, img_height = 256, 256



print('arguments: ')

# path to file which need to predict		  
img_path = sys.argv[1]
weights = sys.argv[2]
# target_size must be same as model input size
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


KerasModel = KerasModel.KerasModel()
model = KerasModel.getModel(img_width, img_height)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# loading precompiled model
model.load_weights(weights)

preds = model.predict_classes(x)
# prints predics array of four classes
print('Predicted:', preds)
