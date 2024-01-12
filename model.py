import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from google.colab import drive
drive.mount('/content/drive')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomRotation(degrees=(-20, 20)),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/drive/MyDrive/archive (5)/'  

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
}


dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=4),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=4, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)  # Since we have two classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available, using CPU instead.")

model_ft = model_ft.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train() 

        running_loss = 0.0
        running_corrects = 0
        total = 0  

        # Iterate over training data
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

            # Print progress within the current epoch
            print(f"\rTraining batch {total}/{dataset_sizes['train']}", end='', flush=True)


        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print(f'\ntrain Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    print('Testing the model after all epochs...')
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0
    total = 0

    # Iterate over test data
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)

    test_loss = running_loss / dataset_sizes['test']
    test_acc = running_corrects.double() / dataset_sizes['test']

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # Check if test accuracy is the best so far and save the model weights
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

    return model



from PIL import Image

def visualize_model(model, num_images_per_class=50):
    was_training = model.training
    model.eval()
    images_so_far = 0

    num_classes = len(class_names)
    total_images = num_images_per_class * num_classes

    fig = plt.figure(figsize=(50, 100)) 

    # Directories for each class
    class_dirs = [os.path.join(data_dir, 'test', class_name) for class_name in class_names]

    with torch.no_grad():
        for class_dir in class_dirs:
            images = [img for img in os.listdir(class_dir) if img.endswith('.jpg') or img.endswith('.png')][:num_images_per_class]
            for img_name in images:
                if images_so_far >= total_images:
                    break

                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img_tensor = data_transforms['test'](img).unsqueeze(0).to(device)

                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)

                ax = plt.subplot(num_classes, num_images_per_class, images_so_far + 1)  
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[0]]}')
                imshow(img_tensor.cpu().data[0])

                images_so_far += 1

    model.train(mode=was_training)
    plt.show()

def evaluate_model(model, dataloader, criterion):
    model.eval()  

    running_loss = 0.0
    running_corrects = 0
    total = 0  

    # Iterate over data
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    final_loss = running_loss / total
    final_acc = running_corrects.double() / total

    print(f'Test Loss: {final_loss:.4f} Acc: {final_acc:.4f}')

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=20)

model_save_path = '/content/drive/MyDrive/archive (5)/muffin_vs_chihuahua_model3.pth'

torch.save(model_ft.state_dict(), model_save_path)

evaluate_model(model_ft, dataloaders['test'], criterion)

model_path = '/content/drive/MyDrive/archive (5)/muffin_vs_chihuahua_model.pth'

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)  

model_ft.load_state_dict(torch.load(model_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

test_dir = '/content/drive/MyDrive/archive (5)/test'

visualize_model(model_ft)