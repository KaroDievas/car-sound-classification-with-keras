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

img_width, img_height = 503, 376



print('arguments: ')

# path to file which need to predict		  
img_path = sys.argv[2]
weights = sys.argv[1]
# target_size must be same as model input size
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


KerasModel = KerasModel.KerasModel()
model = KerasModel.get_model(img_width, img_height)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# loading precompiled model
model.load_weights(weights)

preds = model.predict_classes(x)
# prints predics array of four classes
print('Predicted:', preds)
