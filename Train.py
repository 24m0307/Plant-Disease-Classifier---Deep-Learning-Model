import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PlantDiseaseDetector:
    def __init__(self, base_path='/kaggle/input/plantdisease/PlantVillage'):
        self.base_path = base_path
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        self.model = None
        self.class_names = None
        
    def prepare_data(self):
        """Prepare training and validation datasets"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2  # 20% for validation
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create training dataset
        self.train_dataset = train_datagen.flow_from_directory(
            self.base_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Create validation dataset
        self.val_dataset = val_datagen.flow_from_directory(
            self.base_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        self.class_names = list(self.train_dataset.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.num_classes} classes:")
        for i, class_name in enumerate(self.class_names):
            print(f"{i}: {class_name}")
        
        return self.train_dataset, self.val_dataset
    
    def create_model(self, model_type='custom'):
        """Create CNN model for plant disease detection"""
        if model_type == 'custom':
            self.model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
                layers.MaxPooling2D(2, 2),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        elif model_type == 'transfer':
            # Using pre-trained MobileNetV2 for transfer learning
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            
            base_model.trainable = False  # Freeze base model
            
            self.model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        return self.model
    
    def train_model(self, epochs=20):
        """Train the model"""
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                monitor='val_loss'
            )
        ]
        
        # Train model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Evaluate model performance"""
        # Get predictions
        val_steps = self.val_dataset.samples // self.val_dataset.batch_size
        predictions = self.model.predict(self.val_dataset, steps=val_steps)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = self.val_dataset.classes[:len(predicted_classes)]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def predict_image(self, image_path):
        """Predict disease for a single image"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.class_names[predicted_class], confidence
    
    def predict_from_url(self, image_url):
        """Predict disease from image URL (like Google Photos)"""
        # Download image from URL
        image = keras.utils.get_file(
            'temp_image.jpg',
            image_url,
            cache_subdir='temp'
        )
        
        return self.predict_image(image)
    
    def save_model(self, filepath='plant_disease_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='plant_disease_model.h5'):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Usage example
def main():
    # Initialize detector
    detector = PlantDiseaseDetector()
    
    # Prepare data
    print("Preparing data...")
    train_ds, val_ds = detector.prepare_data()
    
    # Create model (try both 'custom' and 'transfer')
    print("\nCreating model...")
    model = detector.create_model(model_type='transfer')  # Use transfer learning
    
    # Train model
    print("\nTraining model...")
    history = detector.train_model(epochs=20)
    
    # Plot training history
    detector.plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    detector.evaluate_model()
    
    # Save model
    detector.save_model('plant_disease_model.h5')
    
    # Example prediction
    # Replace with actual image path
    # predicted_class, confidence = detector.predict_image('path/to/test/image.jpg')
    # print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()

# For testing with Google Photos or any web image:
"""
# After training, you can test with web images:
detector = PlantDiseaseDetector()
detector.load_model('plant_disease_model.h5')

# Test with image URL
image_url = "https://example.com/plant_image.jpg"
predicted_class, confidence = detector.predict_from_url(image_url)
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
"""
