import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import pandas as pd
from PIL import Image
import os
import wandb
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torch import nn

def extract_lat_lon(file_name):
    '''Function to extract latitude and longitude from file name'''
    # if number of "-"s in images is 1
    if file_name.count("-") == 1:
        lat = file_name.split('-')[0]
        long = file_name.split('-')[1][:-4]
    elif file_name.count("-") == 2:
        # if lat is the negative one
        if file_name[0] == "-":
            lat = file_name.split('-')[0] + "-" + file_name.split('-')[1]
            long = file_name.split('-')[2][:-4]

        else:
            lat = file_name.split('-')[0]
            long = file_name.split('-')[1] + "-" + file_name.split('-')[2][:-4]

    elif file_name.count("-") == 3:
        lat = file_name.split('-')[0] + "-" + file_name.split('-')[1]
        long = file_name.split('-')[2] + "-" + file_name.split('-')[3][:-4]

    return float(lat), float(long)


class PovertyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.file_names = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        lat, long = extract_lat_lon(self.file_names[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure image is RGB

        labels_idx = self.img_labels[(self.img_labels['lat'] == lat) & (self.img_labels['lon'] == long)]

        if labels_idx.empty:
            print("---------------------")
            print("No label for this image")
            print(f"Lat: {lat}, Long: {long}")
            print("File Name:", self.files_names[idx])
            label = torch.rand(1)
        else:
            label = self.img_labels.iloc[labels_idx.index[0], 1]

        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = Compose([
    Resize((224, 224)),  # Match the ViT input size
    ToTensor(),
])

# Create the Dataset
dataset = PovertyDataset(annotations_file='dhs_clusters_2014.csv', img_dir='images-2014', transform=transform)

from torch.utils.data import DataLoader, random_split

total_count = len(dataset)
train_count = int(0.8 * total_count)  # 80% for training
test_count = total_count - train_count  # 20% for testing

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_count, test_count])

# Initialize DataLoaders for each dataset
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model_type = "CNN" # Toggle

if model_type == "vit":
    # Replace classification layer with regression

    from transformers import ViTForImageClassification, ViTConfig

    # Load pre-trained model configuration
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')

    # Modify the configuration for 1 output (regression)
    config.num_labels = 1

    # Load the model with the modified configuration
    model = ViTForImageClassification(config)

    # Replace the classifier head with a new regression layer
    model.classifier = torch.nn.Linear(model.config.hidden_size, 1)

    # Set the model to training mode
    print("Got to model train")
    model.train()

    print("Using GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Setting up optimizer")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Setting up Loss")
    criterion = torch.nn.MSELoss()


    print("Set up Wandb")
    wandb.init(project="Vision Transformers", entity="hamshoe", name="ViT Run, 10 epochs, lr=1e-4")

    print("Setting up Training loop")
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        for images, labels in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()

            # Forward pass
            outputs = model(images).logits

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = loss.item() / len(train_dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Mean Train Loss: {mean_loss:.4f}')
        wandb.log({"epoch": epoch+1, "mean_train_loss": mean_loss})

    # After training
    model.eval()  # Switch to evaluation mode

    # Evaluate on test data
    mean_test_loss = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()  # Ensure labels are correctly shaped
            outputs = model(images)
            logits = outputs.logits.squeeze()  # Remove any unnecessary dimensions

            # Calculate loss
            loss = criterion(logits, labels)
            mean_test_loss += loss.item()

    # Calculate average loss
    mean_test_loss /= len(test_dataloader.dataset)

    print(f'Mean Test Loss: {mean_test_loss:.4f}')
    wandb.log({"mean_test_loss": mean_test_loss})
    wandb.finish()
    torch.save(model.state_dict(), 'vit_model_30epochs.pth')

elif model_type == "CNN":

    # Load the pretrained ResNet model
    resnet18 = models.resnet50(pretrained=True)

    # Modify the final layer for regression
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 1)  # Output one value for regression

    # Define device, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18.to(device)
    optimizer = optim.Adam(resnet18.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Initialize Weights & Biases
    wandb.init(project="Vision Transformers", entity="hamshoe", name="ResNet 50 Run - lr=1e-4")

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        resnet18.train()
        for images, labels in tqdm(train_dataloader, desc="Training", total=len(train_dataloader)):
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        wandb.log({"epoch": epoch + 1, "mean_train_loss": loss.item() / len(train_dataloader.dataset)})

    # Validation
    resnet18.eval()
    mean_test_loss = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            mean_test_loss += loss.item()

    mean_test_loss /= len(test_dataloader.dataset)
    wandb.log({"mean_test_loss": mean_test_loss})

    wandb.finish()
    print("Training complete")