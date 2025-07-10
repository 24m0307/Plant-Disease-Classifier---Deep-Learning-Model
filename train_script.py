# train_script.py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import argparse
import os

def build_model(num_classes, input_shape=(224, 224, 3)):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base layers initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history, save_path="training_plot.png"):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train Acc")
    plt.plot(history.history['val_accuracy'], label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.savefig(save_path)
    plt.show()

def main(data_dir, batch_size, epochs, model_output):
    # Prepare data
    img_size = (224, 224)

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = train_gen.num_classes
    print(f"🔢 Number of classes: {num_classes}")

    # Build and train model
    model = build_model(num_classes=num_classes, input_shape=img_size + (3,))
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    # Save model
    model.save(model_output)
    print(f"✅ Model saved to: {model_output}")

    # Plot history
    plot_history(history)

# ---- ARGUMENT PARSING ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a plant disease classification model using EfficientNetB0.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--model_output', type=str, default="plant_disease_model.h5", help='Output path for the saved model')

    args = parser.parse_args()
    main(args.data_dir, args.batch_size, args.epochs, args.model_output)
