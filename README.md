# 🌿 Plant Disease Classifier

A deep learning project built using Convolutional Neural Networks (CNN) to classify plant leaf images as **healthy** or **diseased**. The model supports early disease detection to promote smart farming through AI-based solutions.

---

## 🔍 Overview

This notebook demonstrates:

* Data preprocessing and augmentation
* CNN architecture design and training
* Model evaluation on validation data
* Manual prediction on user-provided external images

---

## ⚙️ Model Architecture

A Convolutional Neural Network (CNN) with the following structure:

* 4 Convolutional layers with ReLU activation
* MaxPooling after each conv block
* Flatten layer
* Dense layer with 128 units
* **Output layer with softmax activation** for multi-class classification

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])
```

---

## 📈 Training Configuration

* **Loss Function**: `sparse_categorical_crossentropy`
* **Optimizer**: `adam`
* **Output Activation**: `softmax`
* **Evaluation Metric**: `accuracy`

---

## 🧪 Manual Testing

To test the model:

1. Download an image from the internet that fits one of the classes.
2. Set the image path by assigning the file path of the downloaded image to new_image_path:

```python
# Make a prediction on a new image
new_image_path = 'Your_image_path.jpg  # Replace with the path to your downloaded image
```

---

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/plant-disease-classifier.git
   cd plant-disease-classifier
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook Plant_Disease_Classifier.ipynb
   ```

---

## 📦 Dependencies

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* PIL (Pillow)

---

## 🔮 Future Work

* Extend to multi-class classification
* Deploy the model using FastAPI
* Use transfer learning with pre-trained CNNs

---

## 🤝 Contributing

Contributions are welcome! Open issues or pull requests to improve the model or add new features.

## 🔗 Related Repositories

* 💻 **Web App Deployment** → [Plant Disease Classifier – Web App](https://github.com/emon5369/Plant_Disease_Classifier_web_app)
  
  *React + FastAPI web app for real-time plant disease prediction with image upload and confidence scoring.*
