from common import load_data, evaluate_model, plot_confusion_matrix, plot_roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights

# Set the device to use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, test_loader = load_data('path/to/train', 'path/to/test', batch_size=32)

# Load ResNet50 model, modify to accept 1-channel (grayscale) images
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model = model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)

# Training loop
num_epochs = 50 
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()

    # Print training accuracy and loss after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader):.4f}, Training Accuracy: {100.*correct/total:.2f}%")

# Evaluate the model
accuracy, cm, precision, recall, f1, all_labels, all_outputs = evaluate_model(model, test_loader, device, zero_division=1)

# Print evaluation metrics
print(f"Accuracy: {accuracy}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot confusion matrix
plot_confusion_matrix(cm)

# Plot ROC curve
plot_roc_curve(7, all_labels, all_outputs) 