import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize
import os
from PIL import Image
import time

# Custom dataset class for FER (Facial Emotion Recognition)
class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.data = []

        for label, emotion in enumerate(self.classes):
            emotion_dir = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_dir):
                for filename in os.listdir(emotion_dir):
                    img_path = os.path.join(emotion_dir, filename)
                    if os.path.exists(img_path):
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        for _ in range(3):  
            try:
                image = Image.open(img_path).convert('L')  # Convert image to grayscale
                if self.transform:
                    image = self.transform(image)
                return image, label
            except FileNotFoundError:
                time.sleep(1)  # Retry after a short delay
        raise FileNotFoundError(f"Failed to open {img_path} after multiple attempts.")

# Data transformations for preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.Resize((230, 230)),  # Resize images
    transforms.RandomRotation(15),  # Random rotation for data augmentation
    transforms.RandomCrop(224, padding=8),  # Random crop with padding
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),
])

# Function to load data using DataLoader
def load_data(train_dir, test_dir, batch_size):
    train_dataset = FERDataset(train_dir, transform=transform)
    test_dataset = FERDataset(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# Model evaluation function
def evaluate_model(model, test_loader, device, zero_division=1):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=zero_division)
    
    return accuracy, cm, precision, recall, f1, all_labels, all_outputs

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(num_classes, all_labels, all_outputs):
    all_labels_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], np.array(all_outputs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
