# 🔍 Image Classification using CNN - Cats and dogs

This repository contains the implementation of a Convolutional Neural Network (CNN) model built using TensorFlow for classifying images of cats and dogs. The model achieves an accuracy of 78% on the test dataset.

## 📊 Model and Dataset Overview

This project uses a Convolutional Neural Network (CNN) to classify images as either a cat or a dog. The model is trained on a labeled dataset consisting of images of cats and dogs, and the goal is to correctly predict the class (cat or dog) for new, unseen images. The training dataset contains 8000 images - 4000 of dogs and 4000 of cats. The validation dataset contains 2000 images where 1000 images belong to each class.

<a href="https://www.dropbox.com/scl/fi/ppd8g3d6yoy5gbn960fso/dataset.zip?rlkey=lqbqx7z6i9hp61l6g731wgp4v&e=1&st=gdn6pydw&dl=0">Dataset link</a>

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

- **Training Accuracy**: `85.19%`
- **Test Accuracy**: `80.40%`

## 📦 Using the pre-trained model
The repository includes an h5 file that can be used as a pre-trained model to classify images of cats and dogs. A sample code snippet is given below-
```
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model
model = tf.keras.models.load_model('cat_dog_model.h5')

# Load and preprocess image
img = image.load_img('your_image.jpg', target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
label = 'Dog 🐶' if prediction[0][0] > 0.6 else 'Cat 🐱'
print(f'This image is a: {label}')
```

## ✨ Future Improvements

- <b>Data Augmentation:</b> Increase the dataset size by applying more advanced data augmentation techniques.
- <b>Model Optimization:</b> Experiment with different architectures or advanced models.
- <b>Hyperparameter Tuning:</b> Try using a grid search for tuning hyperparameters to improve model performance.
- <b>Transfer Learning:</b> Fine-tune a pre-trained model to boost accuracy.


## 📌 Conclusion

This project shows how deep learning can be used for customer retention problems in the banking sector. The model performs well on the test data and can be integrated into a larger customer relationship system.

