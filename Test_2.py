# IMMEDIATE TEST SCRIPT - Run this after training your model

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define your paths
MODEL_PATH = 'plant_disease_model.h5'  # Your trained model
TEST_IMAGE_PATH = '/kaggle/input/test-img/late-blight-foliage-tomato.jpg'

# Load model
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Load and preprocess image
print("Loading test image...")
image = Image.open(TEST_IMAGE_PATH).convert('RGB')
print("✅ Image loaded successfully!")

# Display original image
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title('Test Image: Healthy Tomato Leaves')
plt.axis('off')
plt.show()

# Preprocess for model
img_resized = image.resize((224, 224))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
print("Making prediction...")
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# You'll need to update this based on your actual training results
# These class names should match the order from your training
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

predicted_class = class_names[predicted_class_idx]

# Display results
print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)
print(f"Image: Healthy_tomato_leaves_(7871755330).jpg")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print("="*50)

# Parse result
if 'Tomato' in predicted_class and 'healthy' in predicted_class:
    print("🎉 SUCCESS: Model correctly identified as Healthy Tomato!")
    result_status = "CORRECT"
elif 'Tomato' in predicted_class:
    print("🔍 PARTIAL: Identified as Tomato but with disease")
    result_status = "PARTIAL"
else:
    print("❌ INCORRECT: Model misidentified the plant type")
    result_status = "INCORRECT"

print(f"Result Status: {result_status}")

# Show top 3 predictions for more insight
print(f"\nTop 3 Predictions:")
top_3_idx = np.argsort(predictions[0])[-3:][::-1]
for i, idx in enumerate(top_3_idx):
    print(f"{i+1}. {class_names[idx]}: {predictions[0][idx]:.2%}")

# Expected vs Actual
print(f"\n" + "="*50)
print("VALIDATION")
print("="*50)
print(f"Expected: Tomato - Healthy (from filename)")
print(f"Predicted: {predicted_class}")
print(f"Match: {'✅ YES' if 'Tomato' in predicted_class and 'healthy' in predicted_class else '❌ NO'}")
print("="*50)
