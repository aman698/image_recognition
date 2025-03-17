# Face Recognition System on Jetson Nano

# Overview

This project implements a face recognition system on the Jetson Nano using the face_recognition library, dlib, and Flask. The system allows users to upload image datasets through a Flask-based web UI, train a face recognition model, and use it to detect persons in real-time.

# Features

Web-based UI for uploading face datasets

Automated model training using face_recognition and dlib

Real-time face detection and recognition using a Jetson Nano

Flask API for managing dataset and recognition tasks

Technologies Used

# Jetson Nano (Hardware)

Python (Programming Language)

Flask (Web Framework)

face_recognition (Face Detection & Recognition)

dlib (Machine Learning Library for Face Processing)

OpenCV (Computer Vision)

Installation

Prerequisites

Ensure your Jetson Nano is set up with the required dependencies:

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip
pip3 install flask face_recognition dlib opencv-python numpy

Clone the Repository

git clone https://github.com/yourusername/jetson-nano-face-recognition.git
cd jetson-nano-face-recognition

Usage

1. Start the Flask Server

python3 app.py

The server will start at http://localhost:5000/.

2. Upload Images via UI

Navigate to http://localhost:5000/ and upload face images for training.

3. Train the Model

Once images are uploaded, trigger the model training through the UI or API.

4. Run Face Recognition

Use image.py to detect and recognize faces from a live camera feed or images.

python3 image.py

File Structure

jetson-nano-face-recognition/
│── app.py          # Flask server for UI and API
│── image.py        # Face recognition using trained model
│── dataset/        # Folder containing uploaded images
│── models/         # Trained face recognition models
│── static/         # Static files (CSS, JS, etc.)
│── templates/      # HTML templates for UI
│── README.md       # Project documentation

# API Endpoints

POST /upload: Upload images for training

POST /train: Train face recognition model

POST /detect: Detect faces in uploaded images
