# Dog Breed Identification Project

## Overview
In this project, I developed a deep learning model to identify dog breeds using the famous Kaggle dataset. The goal was to achieve high accuracy in breed classification.

## Steps Taken
  1. Dataset Preparation:
      - I used the [Kaggle dataset](https://www.kaggle.com/competitions/dog-breed-identification/data?select=labels.csv), which contains labeled images of various dog breeds.
      - Split the dataset into 80% training and 20% validation sets.
      - Converted the images into batches for efficient training using TensorFlow.

  2. Model Training:
      - Utilized TensorFlow to create a convolutional neural network (CNN) architecture.
      - Trained the model on the training data using Google GPU resources.
      - Monitored training progress and adjusted hyperparameters as needed.

  3. Evaluation and Visualization:
      - Evaluated the model’s performance on the validation set.
      - Visualized results using Matplotlib to understand accuracy, loss, and potential areas for improvement.

## Results
The model predicts probability of different unique breeds, the highest probability we take as the prediction by our model for the final prediction for visiualizing the data. While visiualizing the model I also used matplotlib to plot the different probabilities as predicted by the model.

In the image green means the model as predicted succesfully and red means the prediction is wrong.

![Screenshot 2024-07-12 162206](https://github.com/user-attachments/assets/92945397-7bfa-4cd0-a7c4-d947453fb22f)

Probabilites by the model.

![Screenshot 2024-07-12 162218](https://github.com/user-attachments/assets/2470e17c-bc68-4740-8d53-894c345f648d)

