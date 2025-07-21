# 🔍 Image Classification using CNN - Cats and dogs

This repository contains the implementation of a Convolutional Neural Network (CNN) model built using TensorFlow for classifying images of cats and dogs. The model achieves an accuracy of 78% on the test dataset.

## 📊 Model and Dataset Overview

This project uses a Convolutional Neural Network (CNN) to classify images as either a cat or a dog. The model is trained on a labeled dataset consisting of images of cats and dogs, and the goal is to correctly predict the class (cat or dog) for new, unseen images. The training dataset contains 8000 images - 4000 of dogs and 4000 of cats. The validation dataset contains 2000 images where 1000 images belong to each class.

## ⚙️ Technologies Used

- Python
- Google Colab
- TensorFlow / Keras (for CNN)
- Scikit-learn
- PIL

## 🧠 Model Architecture

The CNN model consists of several layers, including convolutional layers, max-pooling layers, and fully connected layers.

- Input Layer: Accepts images resized to 64x64 pixels.
- Convolutional Layers: Multiple convolutional layers with ReLU activation to extract features from the image.
- Dropout Layers: Dropout layers to remove model biasedness deactivating some of the neurons.
- Max-Pooling Layers: Pooling layers to reduce the spatial dimensions of the feature maps.
- Fully Connected Layers: Dense layers at the end for classification (2 classes: cat and dog).
- Output Layer: A sigmoid layer that outputs the probability of each class (cat or dog).


## 📈 Model Performance

- **Training Accuracy**: `85.15%`
- **Test Accuracy**: `78.50%`

## ✨ Future Improvements

- <b>Data Augmentation:</b> Increase the dataset size by applying more advanced data augmentation techniques.
- <b>Model Optimization:</b> Experiment with different architectures or advanced models.
- <b>Hyperparameter Tuning:</b> Try using a grid search for tuning hyperparameters to improve model performance.
- <b>Transfer Learning:</b> Fine-tune a pre-trained model to boost accuracy.


## 📌 Conclusion

This project shows how deep learning can be used for customer retention problems in the banking sector. The model performs well on the test data and can be integrated into a larger customer relationship system.

