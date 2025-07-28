# wound_detection.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import tqdm
# Corrected wound types based on your dataset folder names
WOUND_TYPES = [
    'Abrasions', 'Bruises', 'Burns', 'Cut', 'Diabetic Wounds',
    'Laseration', 'Normal', 'Pressure Wounds', 'Surgical Wounds', 'Venous Wounds'
]

# Path configuration
WOUND_DATA_PATH = r"C:\Users\Aoun Haider\Downloads\python codes VS\New Medical Bot\Wound_dataset copy"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "wound_model.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom dataset class
class WoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self._load_images()
        
    def _load_images(self):
        images = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_path, img_name)
                    images.append((img_path, self.class_to_idx[cls]))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Simple CNN model
class SimpleWoundCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleWoundCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_datasets():
    """Create train, validation, and test datasets"""
    full_dataset = WoundDataset(WOUND_DATA_PATH, transform=transform)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset

def train_model():
    """Train the wound classification model"""
    print(f"Training model for {len(WOUND_TYPES)} wound types...")
    
    # Create datasets and data loaders
    train_dataset, val_dataset, test_dataset = create_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model, loss, and optimizer
    model = SimpleWoundCNN(len(WOUND_TYPES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training variables
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    num_epochs = 30
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # Training phase
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        # Calculate training loss
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model with val accuracy: {val_accuracy:.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, Time: {epoch_time:.1f}s")
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = test_correct / test_total
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)

def plot_training_history(train_losses, val_losses, val_accuracies):
    """Visualize training progress"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

def analyze_wound(image_path):
    """Classify a wound image using trained model"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Load model
    model = SimpleWoundCNN(len(WOUND_TYPES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
    
    # Generate report
    severity = "Severe" if confidence > 0.8 else ("Moderate" if confidence > 0.5 else "Mild")
    
    return {
        'wound_type': WOUND_TYPES[predicted_idx],
        'confidence': confidence,
        'severity': severity,
    }

if __name__ == "__main__":
    train_model()
    
    # Test classification after training
    test_image = os.path.join(WOUND_DATA_PATH, "Abrasions", "example.jpg")  # Replace with actual test image
    if os.path.exists(test_image):
        print("\nTesting classification:")
        result = analyze_wound(test_image)
        print(f"Predicted: {result['wound_type']} ({result['confidence']:.2%})")
        print(f"Severity: {result['severity']}")