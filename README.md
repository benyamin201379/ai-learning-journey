# AI Learning Journey 🚀

A structured public journey documenting my progress toward becoming an AI Engineer through hands-on projects in Deep Learning, Computer Vision, and AI Systems using Python and PyTorch.

---

# About This Repository

This repository contains daily AI engineering projects built from scratch while learning:

- Deep Learning
- Neural Networks
- Computer Vision
- Transfer Learning
- AI System Design
- Real-world AI Applications

The objective is not only to study AI theory but also to build practical AI systems consistently and document the entire learning process publicly.

---

# Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- PIL
- VS Code
- Git
- GitHub

---

# Learning Timeline

---

## Day 01 — First Neural Network

Built my first neural network using PyTorch.

The model learned the relationship:

```python
y = 2x
```

### Concepts Learned

- Tensors
- Neural Networks
- Weights and Bias
- Loss Functions
- Gradient Descent
- Backpropagation

### File

```text
linear_regression.py
```

---

## Day 02 — Nonlinear Neural Network

Built a nonlinear neural network capable of learning:

```python
y = x²
```

### Concepts Learned

- Hidden Layers
- ReLU Activation
- Adam Optimizer
- Nonlinear Function Approximation
- Feature Learning

### File

```text
nonlinear_nn.py
```

---

## Day 03 — Student Score Predictor

Created a regression model that predicts exam scores from study hours.

### Example

Input:

```text
Study Hours
```

Output:

```text
Predicted Exam Score
```

### Concepts Learned

- Regression Problems
- Features and Targets
- Dataset Preparation
- Neural Network Workflow
- Prediction Systems

### File

```text
student_score_predictor.py
```

---

## Day 04 — MNIST Image Classifier

Built my first computer vision model using the MNIST handwritten digits dataset.

The model classifies handwritten digits from:

```text
0 - 9
```

### Concepts Learned

- Computer Vision Basics
- Image Datasets
- DataLoader
- Flatten Layers
- Multi-class Classification
- CrossEntropyLoss

### File

```text
mnist_classifier.py
```

---

## Day 05 — CNN on MNIST

Implemented my first Convolutional Neural Network (CNN).

The CNN improved image classification performance on handwritten digits.

### Concepts Learned

- Convolutional Layers
- Max Pooling
- Feature Extraction
- CNN Architectures
- Spatial Feature Learning

### File

```text
cnn_mnist.py
```

---

## Day 06 — CIFAR-10 CNN Classifier

Trained a CNN on the CIFAR-10 dataset for real-world image classification.

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Features

- Prediction Visualization
- Confidence Scores
- Softmax Probabilities
- RGB Image Classification

### Concepts Learned

- Multi-channel Image Tensors
- RGB Images
- Softmax
- Probability Distributions
- Real-world Datasets

### Files

```text
cifar10_classifier.py
cifar10_viewer.py
```

---

## Day 07 — ResNet18 Image Classifier 🚀

Built a real-world image classifier using a pretrained ResNet18 model.

The model predicts custom images using transfer learning.

### Features

- Pretrained ResNet18
- Transfer Learning
- Confidence Scores
- Image Visualization
- Real-world Predictions

### Example Prediction

```text
Prediction: golden retriever
Confidence: 79.10%
```

### Concepts Learned

- Transfer Learning
- Pretrained Models
- Image Preprocessing
- Softmax Probabilities
- PIL Image Loading
- AI Inference Pipelines

### Technologies

- PyTorch
- Torchvision
- PIL
- Matplotlib

### File

```text
resnet_classifier.py
```

---

## Day 08 — AI Webcam Classifier 📷

Built a real-time webcam AI application using a pretrained ResNet18 model.

The application continuously captures webcam frames and predicts image classes in real time.

### Features

- Live Webcam Feed
- Real-time Predictions
- Confidence Scores
- Continuous AI Inference
- OpenCV Integration

### Concepts Learned

- Real-time Inference
- Webcam Processing
- Frame-based Prediction
- OpenCV Integration
- AI Deployment Workflow

### Technologies

- Python
- OpenCV
- PyTorch
- Torchvision

### File

```text
webcam_ai.py
```

---

## Day 09 — YOLO Real-Time Object Detection 🎯

Built a real-time object detection system using YOLOv8.

The model detects multiple objects simultaneously and draws bounding boxes around them.

### Features

- Real-time Object Detection
- Bounding Boxes
- Multi-object Recognition
- Webcam Integration
- Live AI Inference

### Concepts Learned

- YOLO Architecture
- Object Detection
- Bounding Boxes
- Multi-object Classification
- Real-time Computer Vision

### Technologies

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch

### File

```text
yolo_detector.py
```

---

## Day 10 — Real-Time Face Detection 👤

Built a real-time face detection system using OpenCV.

The application detects and tracks human faces from a live webcam feed.

### Features

- Real-time Face Detection
- Face Tracking
- Bounding Boxes
- Multiple Face Support
- Live Webcam Processing

### Concepts Learned

- Face Detection
- Haar Cascades
- Object Localization
- Feature-based Detection
- Real-time Computer Vision

### Technologies

- Python
- OpenCV

### File

```text
face_detector.py
```

---

## Day 11 — Face Recognition AI 👤

Built a real-time face recognition system using OpenCV and the face_recognition library.

The system identifies whether a detected face matches a known reference image and labels the person accordingly.

### Features

- Real-time webcam face recognition
- Known face identification
- Unknown face detection
- Face encoding generation
- Face similarity matching
- Live bounding boxes
- Name labels on detected faces

### Example

Known Person:

```text
Benyamin

# Repository Structure

```text
ai-learning/
│
├── linear_regression.py
├── nonlinear_nn.py
├── student_score_predictor.py
├── mnist_classifier.py
├── cnn_mnist.py
├── cifar10_classifier.py
├── cifar10_viewer.py
├── resnet_classifier.py
├── webcam_ai.py
├── yolo_detector.py
├── face_detector.py
├── test_image.png
├── yolov8n.pt
└── README.md
```

---

# Current Focus

Current learning areas:

- Deep Learning
- Computer Vision
- CNN Architectures
- Transfer Learning
- Object Detection
- Real-Time AI Systems
- AI Engineering Workflows

---

# Future Roadmap

## Computer Vision

- Face Recognition
- Image Segmentation
- Instance Segmentation
- Pose Estimation
- Tracking Systems

## Deep Learning

- Advanced CNNs
- Transformers
- Attention Mechanisms
- Diffusion Models

## LLMs & AI Systems

- AI Chatbots
- Retrieval-Augmented Generation (RAG)
- AI Agents
- Multi-Agent Systems
- LLM Applications

## Engineering AI

- Aerospace AI
- Scientific Machine Learning
- Simulation-based AI
- Physics-Informed Neural Networks

---

# Goal

The purpose of this repository is to build a strong practical foundation in AI Engineering by creating projects consistently, documenting progress publicly, and gradually moving from simple neural networks to production-ready AI systems.
