# car-sound-classification-with-keras
Car engine sound classification with Keras Deep learning library using spectro analysis pictures of sounds

# Purpose
- To detect concrete car engine sound.

# About
The goal is to create deep learning algorimth which can detect concrete car engine sound. This library it's still in developing. In the data folder you can find four cars which's engines sound was converted to spectro analysis pictures and passed to keras deep learning network. There is a three procedures:
- image_classificator_3_layers.py - with 3 convolutional layers
- image_classificator_more_layers.py - with 5 convolutional layers

Also there is procedure which uses only the saved model from above mentioned files
- predictor.py

Pictures was generated using Wavelet sound explorer http://stevehanov.ca/wavelet/

# Possible usage
- Automatic garden doors system using car engine sound system

# Requirements
- Ubuntu 16.04 LTS
- Keras
- Python 2.7

# Instalation
- Install python 2.7
- Install Keras 
- Install Theano or TensorFlow (I runned it on TensorFlow CPU and GPU versions)

# Running
For training and saving model:
- python image_classificator_mode_layers.py

For predicting with non trained data:
- python predictor.py [path to file which you want to predict]

Prediction example:
- python predictor.py data/experiment/pictures/Balsas087-audi.PNG

Result:
-  ('Predicted:', array([[ 0.,  0.,  1.,  0.]], dtype=float32))

Regarding class dictionary:
- ('class dictionary', {'suzuki': 3, 'kia': 1, 'nissan': 2, 'audi': 0})
- Prediction failed. Because in this audio stream was hearing other car engine sound.


