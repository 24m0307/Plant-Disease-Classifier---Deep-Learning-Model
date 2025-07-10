import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

class QuickPlantTester:
    def __init__(self, model_path='plant_disease_model.h5'):
        """Load trained model and class names"""
        self.model = keras.models.load_model(model_path)
        
        # Define class names (update based on your actual dataset structure)
        self.class_names = [
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
        
        self.img_height = 224
        self.img_width = 224
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image
        image = image.resize((self.img_width, self.img_height))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_from_file(self, image_path):
        """Predict disease from local image file"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        processed_img = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name
        predicted_class = self.class_names[predicted_class_idx]
        
        # Display results
        self.display_prediction(image, predicted_class, confidence)
        
        return predicted_class, confidence
    
    def predict_from_url(self, image_url):
        """Predict disease from image URL (Google Photos, etc.)"""
        try:
            # Download image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Preprocess
            processed_img = self.preprocess_image(image)
            
            # Predict
            predictions = self.model.predict(processed_img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get class name
            predicted_class = self.class_names[predicted_class_idx]
            
            # Display results
            self.display_prediction(image, predicted_class, confidence)
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Error processing image from URL: {e}")
            return None, None
    
    def display_prediction(self, image, predicted_class, confidence):
        """Display image with prediction results"""
        plt.figure(figsize=(10, 6))
        
        # Display image
        plt.imshow(image)
        plt.axis('off')
        
        # Parse class name for better display
        plant_type, disease = self.parse_class_name(predicted_class)
        
        # Title with results
        plt.title(f'Plant: {plant_type}\nDisease: {disease}\nConfidence: {confidence:.2%}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Plant Type: {plant_type}")
        print(f"Disease/Status: {disease}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw Class: {predicted_class}")
        print(f"{'='*50}")
    
    def parse_class_name(self, class_name):
        """Parse class name to extract plant type and disease"""
        if 'Pepper' in class_name:
            plant_type = 'Bell Pepper'
            if 'healthy' in class_name:
                disease = 'Healthy'
            elif 'Bacterial_spot' in class_name:
                disease = 'Bacterial Spot'
            else:
                disease = 'Unknown'
                
        elif 'Potato' in class_name:
            plant_type = 'Potato'
            if 'healthy' in class_name:
                disease = 'Healthy'
            elif 'Early_blight' in class_name:
                disease = 'Early Blight'
            elif 'Late_blight' in class_name:
                disease = 'Late Blight'
            else:
                disease = 'Unknown'
                
        elif 'Tomato' in class_name:
            plant_type = 'Tomato'
            if 'healthy' in class_name:
                disease = 'Healthy'
            elif 'Bacterial_spot' in class_name:
                disease = 'Bacterial Spot'
            elif 'Early_blight' in class_name:
                disease = 'Early Blight'
            elif 'Late_blight' in class_name:
                disease = 'Late Blight'
            elif 'Leaf_Mold' in class_name:
                disease = 'Leaf Mold'
            elif 'Septoria_leaf_spot' in class_name:
                disease = 'Septoria Leaf Spot'
            elif 'Spider_mites' in class_name:
                disease = 'Spider Mites (Two-spotted)'
            elif 'Target_Spot' in class_name:
                disease = 'Target Spot'
            elif 'YellowLeaf_Curl_Virus' in class_name:
                disease = 'Yellow Leaf Curl Virus'
            elif 'mosaic_virus' in class_name:
                disease = 'Mosaic Virus'
            else:
                disease = 'Unknown'
        else:
            plant_type = 'Unknown'
            disease = 'Unknown'
            
        return plant_type, disease
    
    def test_multiple_images(self, image_paths):
        """Test multiple images at once"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\nTesting image {i+1}/{len(image_paths)}: {image_path}")
            predicted_class, confidence = self.predict_from_file(image_path)
            results.append((image_path, predicted_class, confidence))
        
        return results

# Usage examples
if __name__ == "__main__":
    # Initialize tester (make sure your model is trained and saved)
    tester = QuickPlantTester('plant_disease_model.h5')
    
    # Test with local image
    # predicted_class, confidence = tester.predict_from_file('path/to/your/image.jpg')
    
    # Test with Google Photos or any web image
    # Replace with actual image URL
    # image_url = "https://example.com/plant_image.jpg"
    # predicted_class, confidence = tester.predict_from_url(image_url)
    
    # Test multiple images
    # image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = tester.test_multiple_images(image_paths)
    
    print("Quick tester initialized. Use methods to test your images!")

# Example usage in Jupyter notebook:
"""
# Create tester instance
tester = QuickPlantTester('plant_disease_model.h5')

# Test with web image
image_url = "https://your-google-photo-link.jpg"
predicted_class, confidence = tester.predict_from_url(image_url)

# Test with local image
predicted_class, confidence = tester.predict_from_file('test_image.jpg')
"""
