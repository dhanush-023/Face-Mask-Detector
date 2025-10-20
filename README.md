# 🩺 Real-Time Face Mask Detector using CNN & OpenCV

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 📖 Overview

This project is a *Real-Time Face Mask Detection System* built using *Convolutional Neural Networks (CNNs)* and *OpenCV*.
It detects whether a person is wearing a *face mask or not* in real time via a webcam feed.

The model is trained on a dataset of masked and unmasked face images, achieving high accuracy in distinguishing between the two classes.
The live video feed is processed frame by frame, and predictions are displayed with bounding boxes and labels.

---

## 🚀 Features

* 🧠 *Deep Learning CNN model* trained from scratch using TensorFlow/Keras
* 🎥 *Real-time detection* using OpenCV and Haarcascade classifier
* 🎯 *Binary classification:* Mask vs No Mask
* 💾 *Custom dataset support* via Keras ImageDataGenerator
* ⚡ *Lightweight and efficient* — runs on CPU or GPU

---

## 🧩 Tech Stack

| Component        | Technology                 |
| ---------------- | -------------------------- |
| Language         | Python                     |
| Deep Learning    | TensorFlow / Keras         |
| Computer Vision  | OpenCV                     |
| Model Type       | CNN (Sequential)           |
| Dataset Handling | Keras ImageDataGenerator |
| Face Detection   | Haar Cascade Classifier    |

---

## 📂 Project Structure


📁 Real-Time-Face-Mask-Detector
├── 📄 README.md
├── 🧠 mask_detector_model.h5          # Trained CNN model
├── 📁 data/                           # Dataset (mask / no_mask folders)
├── 📄 train_model.py                  # CNN model training script
├── 📄 detect_mask_live.py             # Real-time mask detection using webcam
└── 📄 requirements.txt                # Required dependencies


---

## ⚙ Installation & Setup

### 1️⃣ Clone this Repository

bash
git clone https://github.com/yourusername/real-time-face-mask-detector.git
cd real-time-face-mask-detector


### 2️⃣ Install Dependencies

bash
pip install tensorflow opencv-python numpy


*(You can also use the requirements.txt if provided.)*

### 3️⃣ Prepare Dataset

Your dataset folder structure should look like this:


data/
│
├── with_mask/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── without_mask/
    ├── img1.jpg
    ├── img2.jpg
    └── ...


Update the dataset_path variable in the code to match your local dataset path.

---

## 🧠 Model Architecture

The CNN model is designed for binary image classification:

| Layer Type                | Output Shape   | Parameters |
| ------------------------- | -------------- | ---------- |
| Conv2D (32 filters, 3x3)  | (254, 254, 32) | 896        |
| MaxPooling2D              | (127, 127, 32) | 0          |
| Conv2D (64 filters, 3x3)  | (125, 125, 64) | 18496      |
| MaxPooling2D              | (62, 62, 64)   | 0          |
| Conv2D (128 filters, 3x3) | (60, 60, 128)  | 73856      |
| MaxPooling2D              | (30, 30, 128)  | 0          |
| Flatten                   | (115200)       | 0          |
| Dense (128)               | (128)          | 14745728   |
| Dropout (0.5)             | —              | 0          |
| Dense (1, sigmoid)        | (1)            | 129        |

*Loss Function:* Binary Crossentropy
*Optimizer:* Adam
*Metric:* Accuracy

---

## 🧪 Training

Run the following command to train the model:

bash
python train_model.py


This script:

* Loads images using ImageDataGenerator
* Splits data into training & validation (80/20)
* Trains the CNN for multiple epochs
* Saves the trained model as mask_detector_model.h5

---

## 👁 Real-Time Detection

Run this script to start webcam-based mask detection:

bash
python detect_mask_live.py


*Controls:*

* Press **q** to quit the webcam window.

---

## 📸 Sample Output

| With Mask                                                       | Without Mask                                                     |
| --------------------------------------------------------------- | ---------------------------------------------------------------- |
| ![Mask](https://img.icons8.com/emoji/96/green-circle-emoji.png) | ![No Mask](https://img.icons8.com/emoji/96/red-circle-emoji.png) |

Bounding boxes are drawn around detected faces:

* 🟩 *Green Box:* Person wearing a mask
* 🟥 *Red Box:* Person not wearing a mask

---

## 🧰 Haar Cascade Face Detection

The script uses OpenCV’s pre-trained Haar Cascade Classifier:

python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


It detects faces in each frame before passing them to the CNN model for classification.

---

## 💡 Future Improvements

* [ ] Add MobileNetV2 or Inception-based transfer learning for higher accuracy
* [ ] Deploy using Flask or Streamlit web app
* [ ] Integrate mask detection with attendance systems
* [ ] Support multi-person detection in crowded scenes

---

## 🧑‍💻 Author

*Dhanush P*
💼 *AI/ML Engineer*
🎯 Focus: Deep Learning | Computer Vision | Artificial Intelligence | Model Deployment

---

## 📜 License

This project is released under the *MIT License* — feel free to modify and distribute.

---

## ⭐ Acknowledgments

* TensorFlow and Keras documentation
* OpenCV library
* Haarcascade pretrained model
* Inspiration from real-world COVID-19 safety systems

---

> “AI is not just about machines learning — it’s about humans teaching them to see.”
