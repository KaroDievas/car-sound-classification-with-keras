from setuptools import setup, find_packages
setup(
    version='1.0',
    name='CarSoundClassificationWithKeras',
    packages=find_packages(),
    author='Donatas Kurapkis',
    description='Car sound classification with keras',
    url='https://github.com/KaroDievas/car-sound-classification-with-keras',
    license='MIT',
    install_requires=[
    	'keras',
	'pydub',
	'theano',
	'tensorflow',
	'numpy',
    ],
    python_requires='~=2.7',
)
