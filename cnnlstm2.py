import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def load_frames(video_file):
    vid = cv2.VideoCapture(video_file)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 'L') for ret, frame in iter(vid.read, (False, None)) if ret]
    vid.release()
    return frames


class DrivingDataset(Dataset):
    def __init__(self, labeled_dir, transform=None):
        self.labeled_dir = labeled_dir
        self.transform = transform
        self.video_files = sorted(file for file in os.listdir(labeled_dir) if file.endswith('.hevc'))
        self.label_files = sorted(file for file in os.listdir(labeled_dir) if file.endswith('.txt'))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = os.path.join(self.labeled_dir, self.video_files[idx])
        label_file = os.path.join(self.labeled_dir, self.label_files[idx])
        frames = [self.transform(frame) for frame in load_frames(video_file)] if self.transform else load_frames(video_file)
        labels = torch.tensor(np.loadtxt(label_file), dtype=torch.float32)
        return frames, labels

def collate_fn(batch):
    frames, labels = zip(*batch)
    frames = pad_sequence([torch.stack(frame) for frame in frames], batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    return frames, labels

class CNNLSTM(nn.Module):
    def __init__(self, frame_height, frame_width):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * frame_height//2//2 * frame_width//2//2, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size * timesteps, -1)
        x = self.fc1(x)
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        return x

def validate_model(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return running_loss / i

def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate_model(model, criterion, val_dataloader, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/i:.4f}, Validation Loss: {val_loss:.4f}')

    return model

def predict(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_outputs = [model(inputs.to(device)) for inputs in dataloader]
    return torch.cat(all_outputs)

# Define transformations
frame_height, frame_width = 64, 64
transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((frame_height, frame_width)), transforms.ToTensor()])


# Create dataset and data loaders
train_dataset = DrivingDataset('labeled/labeled1/', transform=transform)
val_dataset = DrivingDataset('labeled/val1/', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Instantiate the model
model = CNNLSTM(frame_height, frame_width)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Move the model and loss function to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

# Train the model
model = train_model(model, criterion, optimizer, train_dataloader, val_dataloader, device, num_epochs=25)
