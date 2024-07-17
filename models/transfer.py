from common import load_data, evaluate_model, plot_confusion_matrix, plot_roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.nn import init

# Define a modified ResNet model for feature extraction
class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)

# Define a Masked Auto-Dropout (MAD) module
class MAD(nn.Module):
    def __init__(self, drop_rate):
        super(MAD, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            binary_mask = torch.bernoulli(torch.full(x.size(), 1 - self.drop_rate)).to(x.device)
            return x * binary_mask
        return x

# Define the TransFER model integrating CNN and Transformer
class TransFER(nn.Module):
    def __init__(self, num_classes=7):
        super(TransFER, self).__init__()
        self.stem_cnn = ModifiedResNet()
        self.local_cnn1 = nn.Conv2d(512, 2048, kernel_size=3, padding=1)
        self.local_cnn2 = nn.Conv2d(512, 2048, kernel_size=3, padding=1)
        self.mad = MAD(drop_rate=0.6)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True),  
            num_layers=8
        )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem_cnn(x)
        x1 = self.local_cnn1(x)
        x2 = self.local_cnn2(x)
        x1 = self.mad(x1)
        x2 = self.mad(x2)
        x = torch.cat((x1, x2), dim=2)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.transformer_encoder(x)
        x = self.classifier(x.mean(dim=1))
        return x

def main():
    # Set up the device, model, loss function, optimizer, and learning rate scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransFER(num_classes=7).to(device)
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
