# Car sound classification with Keras
Car engine sound classification with Keras Deep learning library using Morlet pictures of sounds

# Purpose
- To detect concrete car engine sound.

# About
The goal is to create deep learning algorithm which can detect concrete car engine sound. This library still in developing.
In later versions there are no provided training data due huge amount of it.
In Data/audio/original_source you can find m4a files of different cars engines. Later I will provide file with representation.
In order to use that data you need to convert it to wav files and pass to generate train data.

# Possible usage
- Automatic garden doors system using car engine sound system

# Requirements
- Anaconda 3
- Keras
- Python 3.6
- pip >= 9.*
- Librosa
- ffmpeg
- librosa
- tensorflow
- matplotlib

# Instalation
- python setup.py install --user

# Usage

In forlder Data/raw you can put your data in separated in folders.
For example:
Data/raw/audi - some wav files
Data/raw/other - some wav files

You can have as many as you want categories. Just do not forget to modify Models/KerasModel.py, TrainModel.py and Predictor.py cause these are configurated for binary usage.

For data generation from wav file:
- python Main.py
It will generate all needed structure for you.
- Slices audio source (currently only mono)
- Puts half sliced sources to train half to validation directories


For training network:
- python TrainModel.py
Before running update these parameter by your needs:
- nb_train_samples
- nb_validation_samples
- nb_epoch
- batch_size
Only improved weights are saving.
At the train end you will have two tables about how gone you training.

For prediction:
- python Predictor.py path/to/your/weight_file path/to/image_you_want_to_predict

# Conclusion
It better to use binary classification (only two classes) due to you can concentrate more train data and model will be more accurate. For me it gives ~0.8 accuracy or ~0.2 loss. I had around 3700 pictures for each class in validation and train.

I have tried to it by classes. Had 47 classes each class has about 100-200 pictures so it's really small amount and network trains very slow.
Results was very poor. This will give me only ~4.1 accuracy which is basically nothing. So for this model need more data or do something else.