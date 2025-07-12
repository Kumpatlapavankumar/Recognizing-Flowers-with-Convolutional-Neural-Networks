# 🌸 Flower Classification using Convolutional Neural Networks (CNN)

This project focuses on classifying different types of flowers using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The goal is to recognize and classify images of flowers into 5 categories: Daisy, Dandelion, Rose, Sunflower, and Tulip. This deep learning model is trained on the [Flowers Recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) and uses data augmentation techniques to improve performance.

---

## 📂 Dataset Overview

- **Source**: Kaggle - [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- **Classes**:
  - Daisy
  - Dandelion
  - Rose
  - Sunflower
  - Tulip
- **Format**: JPEG images
- **Total Size**: ~600 MB

---

## 🔧 Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `TensorFlow`, `Keras`
  - `OpenCV` (for image preprocessing)
  - `Matplotlib`, `Seaborn` (for visualizations)
  - `NumPy`, `Pandas` (for data handling)

---

## 🧠 Model Architecture

```text
Input Image (180x180x3)
↓
Conv2D (32 filters) → ReLU → MaxPooling2D
↓
Conv2D (64 filters) → ReLU → MaxPooling2D
↓
Conv2D (128 filters) → ReLU → MaxPooling2D
↓
Flatten
↓
Dense (128 units) → ReLU
↓
Dropout (0.5)
↓
Dense (5 units) → Softmax (Output)
