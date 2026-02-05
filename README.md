# 🦴 Bone Fracture Detection System

An end-to-end **AI-powered medical imaging application** that analyzes X-ray images to detect bone fractures, provides confidence-based predictions, and generates **professional, downloadable PDF medical reports** through a clean, patient-friendly web interface.
- live url here you can visit and test my project
- [Active link](https://bone-fracture-detection-app-gzlsnihz3ynev2odzzju58.streamlit.app/#bone-fracture-detection-system)
---

## 🚨 Problem Statement

Bone fractures are one of the most common injuries worldwide. In many regions:

* Access to expert radiologists is limited
* Manual X-ray interpretation is time-consuming
* Early-stage or subtle fractures are often misdiagnosed

This creates delays in treatment and increases the risk of complications.

👉 **Goal:** Build an AI-assisted screening tool that can help detect bone fractures quickly, consistently, and accessibly.

---

## 🎯 Project Objectives

* Automatically detect bone fractures from X-ray images
* Provide clear **Normal / Fractured** predictions
* Display prediction confidence to improve transparency
* Generate a **professional medical-style PDF report**
* Deliver a clean, easy-to-use patient-facing interface
* Deploy the system as a live web application

---

## 🧠 Solution Overview

This project uses a **Convolutional Neural Network (CNN)** trained on labeled X-ray images to classify fractures. The trained model is integrated into a **Streamlit web application**, allowing users to upload X-ray images and receive instant results.

The system is designed for:

* Educational use
* Research demonstrations
* AI-assisted screening (not a medical diagnosis replacement)

---

## 🏗️ Project Architecture

```
Bone_Fracture_Detection_CNN/
│
├── 01_Data/               # Dataset (train / test)
├── 02_Notebooks/          # Phase-wise development notebooks
├── 03_Models/             # Trained CNN model (.h5)
├── 04_Inference/          # Inference & prediction logic
├── 05_UI/                 # Streamlit web application (app.py)
├── 06_Reports/            # Generated PDF reports
├── README.md              # Project documentation
```

---

## 🔬 Model Details

* **Architecture:** Convolutional Neural Network (CNN)
* **Input Size:** 224 × 224 RGB X-ray images
* **Output:** Binary classification (Fractured / Normal)
* **Activation:** Sigmoid
* **Loss Function:** Binary Crossentropy
* **Framework:** TensorFlow / Keras

The model outputs a probability score which is converted into a **confidence percentage** for better interpretability.

---

## ⚙️ Technology Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** NumPy, Pillow (PIL)
* **Web Framework:** Streamlit
* **Report Generation:** ReportLab
* **Deployment:** Streamlit Cloud
* **Version Control:** Git & GitHub

---

## 🖥️ Application Features

### ✅ User Interface

* Upload X-ray image (JPG / PNG)
* Instant prediction result
* Confidence score display
* Clean and patient-friendly layout

### 📄 PDF Medical Report

Each report includes:

* Prediction result
* Confidence percentage
* Timestamp
* AI disclaimer
* Project branding

Reports are downloadable directly to the user’s device.

### 🧾 Branding

* Footer and reports include:

  > © SHUBHAM MADDHESIYA

---

## 📸 Sample Workflow

1. Upload an X-ray image
2. Model analyzes the image
3. Prediction displayed (Normal / Fractured)
4. Confidence score shown
5. Download detailed PDF report

---

## ⚠️ Disclaimer

> This system is intended for **educational and research purposes only**.
> It is **not a substitute for professional medical diagnosis or treatment**.
> Always consult a qualified healthcare professional.

---

## 🚀 Live Application

🔗 **Live Demo:** *(Add Streamlit Cloud link here)*

🎥 **Demo Video:** *(Optional – YouTube unlisted link)*

---

## 📈 Achievements & Learnings

* Built a complete AI pipeline from training to deployment
* Applied CNNs to real-world medical imaging problems
* Learned end-to-end ML product development
* Gained experience in Streamlit deployment
* Implemented automated PDF report generation

---

## 🧑‍💻 Author

**Shubham Maddhesiya**
AI & Deep Learning Enthusiast

---

## ⭐ If you found this project useful

Give it a ⭐ on GitHub and feel free to connect!
