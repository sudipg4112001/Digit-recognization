# Digit Recognition through webcam and CNN
![Python](https://img.shields.io/badge/Python%20-Python%203.9.1-yellowgreen?style=for-the-badge&logo=python)
![Tensorflow](https://img.shields.io/badge/Tensorflow%20-blue?style=for-the-badge&logo=Tensorflow)
![Sklearn](https://img.shields.io/badge/Sklearn%20-green?style=for-the-badge&logo=Sklearn)
![OpenCV](https://img.shields.io/badge/OpenCV%20-important?style=for-the-badge&logo=OpenCV)
![Numpy](https://img.shields.io/badge/Numpy%20-blue?style=for-the-badge&logo=Numpy)
- Dataset: `MNIST Dataset`
- Deep Learning Algorithm used: `Convolutional Neural Network(CNN)`

For training our CNN model, the MNIST dataset was compiled with images of digits from various scanned documents and then normalized in size. Each image is of a dimension, 28×28 i.e total 784 pixel values.

We do not need to download the dataset from any external source as we will import it from keras.datasets

1. Train a CNN (Convolutional Neural Network) on MNIST dataset, which contains a total of 70,000 images of handwritten digits from 0-9 formatted as 28×28-pixel monochrome images.
2. For this, first split the dataset into train and test data with size 60,000 and 10,000 respectively.
3. Then, preprocess the input data by reshaping the image and scaling the pixel values between 0 and 1.
4. Then design the neural network and train the model, and save it for future use
5. Next, we are going to use a webcam as an input to feed an image of a digit to our trained model.
6. Model will process the image to identify the digit and return a series of 10 numbers corresponding to the ten digits with an activation on the index of the proposed digit.
# Dependencies 
- Keras
- Tensorflow
- OpenCV
- Sklearn
- Numpy

# Exceution
- First run `train.py` to train CNN model using MNIST dataset and save the model.
- Then, run `recognizedigit.py` to produces a prediction displayed at the webcam by users. The model is loaded and appropriate functions are used to capture video via webcam and pass it as input to the saved model.
