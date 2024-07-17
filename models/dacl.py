from common import load_data, evaluate_model, plot_confusion_matrix, plot_roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights

# Define an Attention Module to refine feature representations
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Generate attention weights
        q = self.conv1(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.conv1(x).view(batch_size, -1, height * width)
        v = self.conv2(x).view(batch_size, C, height * width)

        attn_weights = self.softmax(torch.bmm(q, k))
        attn_output = torch.bmm(attn_weights, v.transpose(1, 2)).view(batch_size, C, height, width)

        return x + attn_output

# Define a custom model incorporating the Attention Module and ResNet18
class DACLModel(nn.Module):
    def __init__(self, num_classes):
        super(DACLModel, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first layer to accept 1-channel (grayscale) input
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final FC layer
        self.base_model.fc = nn.Identity()  
        
        self.attention = AttentionModule(in_channels=512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), 512, 1, 1)  # Change shape to [batch_size, 512, 1, 1]
        x = self.attention(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

def main():
    # Set up the device, model, loss function, optimizer, and learning rate scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DACLModel(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)

    # Load data
    train_loader, test_loader = load_data('/content/drive/MyDrive/FacialExpressionRecognition/FER-2013/train', 
                                          '/content/drive/MyDrive/FacialExpressionRecognition/FER-2013/test', 
                                          batch_size=32)

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

if __name__ == "__main__":
    main()
