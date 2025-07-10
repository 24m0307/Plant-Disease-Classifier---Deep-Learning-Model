# test_script.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

# ---- CLASS LABELS ----
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy', 
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ---- IMAGE LOADING AND PREPROCESSING ----
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image, image_array

# ---- PREDICT FUNCTION ----
def predict(model, image_array):
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    return predicted_index, confidence, predictions[0]

# ---- VISUALIZATION ----
def show_image_with_title(image, title="Test Image"):
    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

# ---- MAIN FUNCTION ----
def main(model_path, test_image_path):
    print("🔄 Loading model...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")

    print("📷 Loading test image...")
    raw_image, image_array = preprocess_image(test_image_path)
    print("✅ Image loaded and preprocessed!")

    show_image_with_title(raw_image, title=os.path.basename(test_image_path))

    print("🤖 Running prediction...")
    pred_idx, confidence, full_probs = predict(model, image_array)
    predicted_class = class_names[pred_idx]

    print("\n🧠 Prediction Results")
    print("="*40)
    print(f"Image: {os.path.basename(test_image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("="*40)

    print("\n📊 Top 3 Predictions:")
    top_3 = np.argsort(full_probs)[-3:][::-1]
    for i, idx in enumerate(top_3):
        print(f"{i+1}. {class_names[idx]}: {full_probs[idx]:.2%}")

    print("\n✅ Completed.")

# ---- ARGUMENT PARSING ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained plant disease model on a single image.")
    parser.add_argument('--model', type=str, default='plant_disease_model.h5', help='Path to trained model (.h5)')
    parser.add_argument('--image', type=str, required=True, help='Path to test image (.jpg/.png)')

    args = parser.parse_args()
    main(args.model, args.image)
