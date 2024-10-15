# Fashion - Clothes Classification App using an AI model

This project implements a simple neural network using TensorFlow to classify images of clothing from the Fashion MNIST dataset. The model is trained and tested on 28x28 grayscale images, categorized into 10 different classes, including items such as shirts, sneakers, and bags. The project includes training a model, evaluating its accuracy, and saving it for future predictions. 

## Features
- Image Classification: Classifies uploaded images of clothing into predefined categories.
- User Interface: Provides an easy-to-use UI for uploading images and viewing results.
- Model Training: Utilizes a convolutional neural network for accurate classification.
- Real-Time Predictions: Quickly generates predictions once an image is uploaded.

## Model Architecture
The neural network consists of:
- A Flatten layer to convert the input 28x28 images into a 1D vector.
- A Dense layer with 128 neurons and ReLU activation function.
- An output Dense layer with 10 neurons (one for each class) and a softmax activation function, which provides class probabilities.

## Training
The model is trained on 60,000 images with their corresponding labels and validated on a test set of 10,000 images. It uses the Adam optimizer and sparse categorical cross-entropy as the loss function.
