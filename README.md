# Real-Time Face Detection using TensorFlow & OpenCV

This project implements an end-to-end real-time face detection system built from scratch using TensorFlow, OpenCV, and Albumentations.
It covers the complete machine learning pipeline — from data collection and annotation to model training and real-time inference using a webcam.

# Features

Webcam-based data collection
Manual annotation using LabelMe
Extensive data augmentation

Custom deep learning model with:
  Binary face classification
  Bounding box regression

Custom training loop with masked regression loss
Real-time face detection using OpenCV
CPU-friendly (no GPU required)

# Model Overview
Backbone: VGG16 (ImageNet pretrained)
Input Size: 120 × 120 × 3

Outputs:
  class: Face / No-Face (binary classification)
  bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)

Loss Functions:
  Binary Crossentropy for classification
  Custom localization loss for bounding box regression

Training Strategy:
  Regression loss applied only when a face is present
  Early stopping and checkpointing for stability

# Project Structure
face-detection/
├── data/ # Raw dataset (before augmentation)
│ ├── images/
│ ├── labels/
│ ├── train/
│ ├── test/
│ └── val/
│
├── aug_data/ # Augmented dataset
│ ├── train/
│ ├── test/
│ └── val/
│
├── scripts/
│ ├── collect_images.py # Webcam image collection
│ ├── augment_data.py # Data augmentation pipeline
│ ├── build_dataset.py # tf.data input pipeline
│ ├── train_model.py # Model training
│ └── realtime_face.py # Real-time inference
│
├── models/
│ └── facetracker_functional.keras # Trained model (not tracked in GitHub)
│
├── requirements.txt
├── README.md
└── .gitignore

# Dataset Creation
--Image Collection
Images are captured using the system webcam and stored locally.
python scripts/collect_images.py

# Annotation
Images are annotated manually using LabelMe.
labelme data/images
Each image is labeled with a bounding box around the face.

# Data Augmentation
To improve robustness and reduce overfitting, the dataset is augmented using Albumentations:
  Horizontal flips
  Brightness and contrast adjustment
  Rotation
  Handling both face and no-face images

python scripts/augment_data.py

# Training the Model
The model is trained using a custom tf.keras.Model subclass with:
  Custom train_step
  Masked bounding box loss
  Learning rate decay
  Early stopping

python scripts/train_model.py

The trained functional model is saved as:
models/facetracker_functional.keras

# Real-Time Face Detection
Run the webcam-based inference:
  python scripts/realtime_face.py

Controls:
Press Q or ESC to exit

# Installations
Create a clean environment (recommended):

  conda create -n cvface python=3.9 -y
  conda activate cvface
  pip install -r requirements.txt

# Requirements
tensorflow
opencv-python
albumentations
numpy
matplotlib
labelme

# Results
Stable training on CPU
Accurate face localization in real-time
Robust performance under lighting and orientation changes

# Author
Manish Kumar Gupta
Electronics & Communication Engineering
BIT Mesra
