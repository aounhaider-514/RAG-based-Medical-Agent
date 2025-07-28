import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

# Use raw string for Windows paths
CHEXPERT_PATH = r"C:\Users\Aoun Haider\Downloads\python codes VS\New Medical Bot\chest_xray"
MODEL_PATH = r"./models/xray_model.h5"  # Save trained model here

# Conditions for pneumonia detection
CONDITIONS = ['NORMAL', 'PNEUMONIA']  # Matches your dataset folder names

def preprocess_xray(image_path):
    """Preprocess X-ray image for pneumonia detection"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize and convert to array
    img = img.resize((224, 224))  # DenseNet121 expects 224x224
    img_array = np.array(img) / 255.0
    
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((img_array * 255).astype(np.uint8))
    
    # Convert to 3 channels (DenseNet121 requires RGB)
    img_array = enhanced / 255.0
    return np.stack((img_array,)*3, axis=-1)

def build_xray_model():
    """Create CNN model for pneumonia detection using transfer learning"""
    # Load pre-trained DenseNet121 model
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base layers
    base_model.trainable = False
    
    # Add custom classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(len(CONDITIONS), activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def analyze_xray(image_path):
    """Analyze X-ray image for pneumonia detection"""
    # Preprocess image
    img_array = preprocess_xray(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Make prediction
    prediction = model.predict(img_array)[0]
    predicted_idx = np.argmax(prediction)
    
    return {
        'condition': CONDITIONS[predicted_idx],
        'confidence': float(prediction[predicted_idx]),
        'all_predictions': {CONDITIONS[i]: float(p) for i, p in enumerate(prediction)}
    }

def train_xray_model():
    """Train pneumonia detection model using transfer learning"""
    print("Starting X-ray pneumonia detection training with transfer learning...")
    
    # Prepare dataset using ImageDataGenerator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2  # 20% for validation
    )
    
    # Load images from directory structure
    train_generator = train_datagen.flow_from_directory(
        os.path.join(CHEXPERT_PATH, "train"),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=CONDITIONS,
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        os.path.join(CHEXPERT_PATH, "train"),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes=CONDITIONS,
        subset='validation'
    )
    
    # Build model
    model = build_xray_model()
    
    # Callbacks for training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
    
    # Initial training with frozen base
    print("Phase 1: Training with frozen base layers...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=10,
        callbacks=callbacks
    )
    
    # Fine-tune with unfrozen base layers
    print("Phase 2: Fine-tuning with unfrozen base layers...")
    model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=3
    )
    
    # Save final model
    model.save(MODEL_PATH)
    print(f"Training complete! Model saved to {MODEL_PATH}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2f}")

if __name__ == "__main__":
    train_xray_model()