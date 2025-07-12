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
```

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## 🧪 Data Augmentation

To avoid overfitting and improve model generalization, the following augmentations were applied using `ImageDataGenerator`:

- Rotation
- Zoom
- Width & Height Shift
- Horizontal Flip
- Rescaling

---

## 📊 Visualizations & Analysis

### 1. **Class Distribution**

![Class Distribution](assets/class_distribution.png)

The dataset is slightly imbalanced, with `Dandelion` having the most images.

---

### 2. **Sample Images**

![Sample Images](assets/sample_images.png)

Random samples of each flower class after augmentation.

---

### 3. **Training & Validation Accuracy**

![Training Accuracy](assets/training_accuracy.png)

The model converged well, with training accuracy reaching **~95%** and validation accuracy around **~88%**.

---

### 4. **Training & Validation Loss**

![Training Loss](assets/training_loss.png)

Loss steadily decreased, indicating a stable training process.

---

### 5. **Confusion Matrix**

![Confusion Matrix](assets/confusion_matrix.png)

The model performs best on **Dandelion** and **Sunflower**, with some misclassifications among similar-looking classes like `Daisy` and `Tulip`.

---

## ✅ Final Results

| Metric            | Value        |
|-------------------|--------------|
| Training Accuracy | ~95%         |
| Validation Accuracy | ~88%      |
| Test Accuracy     | ~87%         |

---

## 🚀 How to Run the Project

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/flower-classification-cnn.git
   cd flower-classification-cnn
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Download from [Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) and extract it to the project folder.

4. **Run the notebook**:
   ```bash
   jupyter notebook Recognizing\ Flowers\ with\ Convolutional\ Neural\ Networks.ipynb
   ```

---

## 🔮 Future Improvements

- Implement transfer learning using `EfficientNet`, `ResNet`, or `VGG16`
- Hyperparameter tuning using `KerasTuner` or `Optuna`
- Deploy the model using **Streamlit** or **Flask**
- Convert the model to TensorFlow Lite for mobile applications

---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Kaggle for the flower dataset
- TensorFlow & Keras documentation
