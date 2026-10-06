# 🌱 Plant Disease Detection Using Hybrid AI

An AI-powered plant disease detection system that uses **Deep Learning, Convolutional Neural Networks (CNNs), and Transfer Learning** to identify crop diseases from leaf images.

The system allows users to upload or capture a leaf image and receive an instant disease prediction along with a confidence score and recommendations.

---

## 📌 About the Project

Plant diseases are a major threat to agricultural productivity and food security.

Traditional disease identification generally depends on manual inspection by farmers or agricultural experts. This process can be time-consuming and may lead to incorrect diagnosis, especially when diseases have similar visual symptoms.

This project proposes an **AI-based crop disease detection system** that analyzes plant leaf images and automatically identifies whether the plant is healthy or affected by a particular disease.

The system uses image preprocessing, data augmentation, CNN-based feature extraction, and transfer learning to achieve reliable disease classification.

It is designed as a simple and accessible application that can assist farmers and agricultural professionals with early disease identification.

---

## 🎯 Objectives

- Detect plant diseases automatically from leaf images.
- Reduce dependency on manual disease identification.
- Provide fast and reliable disease predictions.
- Use CNN and Transfer Learning for image classification.
- Improve model performance using data augmentation.
- Provide prediction confidence scores.
- Provide disease prevention and cultivation recommendations.
- Support real-time image-based diagnosis.
- Reduce unnecessary pesticide usage.
- Promote sustainable agricultural practices.

---

## ✨ Key Features

### 🌿 Plant Disease Detection

Identify diseases from plant leaf images using an AI-based classification model.

### 📷 Image Upload

Users can upload a leaf image through the web application.

### 📱 Camera Input

The system can support live camera input for real-time field diagnosis.

### 🤖 AI-Based Classification

The system uses CNN-based deep learning and transfer learning for disease classification.

### 🎯 Confidence Score

The application provides the predicted disease along with the model's confidence level.

### 🔍 Explainable AI

XAI techniques such as **Grad-CAM** can be used to highlight the regions of the leaf that influenced the model's prediction.

### 💡 Recommendations

The application can provide:

- Disease information
- Treatment suggestions
- Preventive measures
- Cultivation tips

### 🌱 Sustainable Agriculture

Accurate disease identification can help reduce unnecessary pesticide usage and crop losses.

---

# 🏗️ System Architecture

```text
                  ┌─────────────────────┐
                  │     Leaf Image      │
                  │ Upload / Camera     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Image Preprocessing │
                  │ Resize + Normalize  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Augmentation   │
                  │ Flip / Rotate / Zoom│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ CNN / MobileNetV2   │
                  │ Feature Extraction  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Disease Classifier  │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        ┌─────────────────┐      ┌─────────────────┐
        │ Disease Label   │      │ Confidence Score│
        └────────┬────────┘      └────────┬────────┘
                 │                        │
                 └────────────┬───────────┘
                              ▼
                   ┌─────────────────────┐
                   │ Recommendation /    │
                   │ Prevention Tips     │
                   └─────────────────────┘
