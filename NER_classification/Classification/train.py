import kagglehub
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision.models as pretrained_models

# Download dataset
path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Path to dataset files:", path)

EPOCHS = 2
BATCH_SIZE = 32
LEARNIG_RATE = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_data = datasets.ImageFolder(path + "/raw-img", transform=transformer)
train_size = int(len(full_data) * 0.8)
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = pretrained_models.resnet50(pretrained=True)
model = model.to(device)

for parameters in model.parameters():
    parameters.requires_grad = False
for parameters in model.fc.parameters():
    parameters.requires_grad = True

num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.fc.parameters(), lr=LEARNIG_RATE)
model = model.to(device)
torch.save(model.state_dict(), './clf_mode')

for epoch in range(EPOCHS):
    model.train()
    running_correct = 0
    running_total = 0
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        if (i + 1) % int(len(train_loader) / 5) == 0:
            step_accuracy = 100 * running_correct / running_total
            print(f"Epoch: {epoch + 1}/{EPOCHS} | Step: {i + 1}/{len(train_loader)} | "
                  f"Train Loss: {loss.item():.4f} | Train Acc: {step_accuracy:.2f}%")
            running_correct = 0
            running_total = 0

    with torch.no_grad():
        model.eval()
        running_correct = 0
        running_total = 0
        for i, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
        test_accuracy = 100 * running_correct / running_total
        print(f"Epoch: {epoch + 1}/{EPOCHS} | Test Acc: {test_accuracy:.2f}%")

# torch.save(model.state_dict(), './clf_model')