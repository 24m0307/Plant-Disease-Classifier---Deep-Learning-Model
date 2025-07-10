# 🌿 Plant Disease Classifier - Deep Learning Model

This repository contains a trained deep learning model for classifying diseases in plant leaves (Tomato, Potato, and Pepper Bell) using **TensorFlow and Keras**.

It supports **15 plant disease classes** including healthy and infected conditions. You can immediately test the model on a new leaf image using the provided test script.

## 📦 Dataset

This project uses the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) available on Kaggle. It contains 15 different classes of plant leaf images including both healthy and diseased conditions of tomato, potato, and pepper bell plants.

The dataset was used to train a deep learning model for disease classification. It includes high-resolution `.JPG` images grouped by class folders.

> Note: The full dataset is **not included** in this repository due to file size restrictions.  
> Please download it directly from [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) if you want to train the model yourself.


## 🧠 Model Classes

This model was trained to classify the following 15 classes:

- Pepper__bell___Bacterial_spot  
- Pepper__bell___healthy  
- Potato___Early_blight  
- Potato___Late_blight  
- Potato___healthy  
- Tomato_Bacterial_spot  
- Tomato_Early_blight  
- Tomato_Late_blight  
- Tomato_Leaf_Mold  
- Tomato_Septoria_leaf_spot  
- Tomato_Spider_mites_Two_spotted_spider_mite  
- Tomato_Target_Spot  
- Tomato_Tomato_YellowLeaf_Curl_Virus  
- Tomato_Tomato_mosaic_virus  
- Tomato_healthy

---

┌─────────────────────────────┐
│  Input Layer (224x224x3)    │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ EfficientNetB0 (frozen)     │  ← Pretrained on ImageNet
│   (Convs + SE Blocks)       │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ GlobalAveragePooling2D      │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Dropout(0.3)                │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Dense(128, relu)            │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Dropout(0.2)                │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│ Dense(15, softmax)          │ ← 15 plant classes
└─────────────────────────────┘

