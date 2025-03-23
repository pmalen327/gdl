import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

# add parent directory to sys.path to fetch the dataloader and model files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import MRI dataset and mri_to_graph from dataloader
from dataloader import MRI_Dataset, mri_to_graph

# import GCN from models.gcn
from models.gcn import GCN

# set device to GPU (CUDA) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
# define path to your dataset (set in .env file)
from dotenv import load_dotenv
load_dotenv()
data_dir = os.getenv('DATA_PATH')

# hyperparameters
batch_size = 4
num_epochs = 20
learning_rate = 0.001
connectivity = 4  # 4 or 8 for graph connectivity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define dataset and dataloader
dataset = MRI_Dataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in dataloader:
    images = batch['image']
    masks = batch['mask']
    
    # move the images and masks to the same device as the model
    images = images.to(device)
    masks = masks.to(device)

# hyperparameters
batch_size = 4
num_epochs = 20
learning_rate = 0.001
connectivity = 4  # 4 or 8 for graph connectivity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# intialize model args
input_dim = 4  # number of channels in MRI data
hidden_dim = 64
output_dim = 3  # number of segmentation classes

# load data
# define path to your dataset (set in .env file)
from dotenv import load_dotenv
load_dotenv()
data_dir = os.getenv('DATA_PATH')

# Define dataset and dataloader
dataset = MRI_Dataset(data_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# initialize model
# move the model to the correct device (CUDA)
model = GCN(in_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        images = batch['image'].to(device)  # shape [B, 4, 240, 240]
        masks = batch['mask'].to(device)  # shape [B, 240, 240]

        batch_loss = 0.0

        for i in range(images.shape[0]):
            print(f"Shape of images[{i}]: {images[i].shape}")
            mri_slice = images[i].to(device)

            if images[i].ndim == 3 and images[i].shape[0] == 4:  # shape (4, 240, 240)
                mri_slice = images[i].permute(1, 2, 0)  # shape [240, 240, 4]
            elif images[i].ndim == 3 and images[i].shape[2] == 4:  # shape (240, 240, 4)
                mri_slice = images[i]
            elif images[i].ndim == 2:  # shape (240, 240) - single channel
                mri_slice = np.expand_dims(images[i], axis=-1)  # add 1 channel
            else:
                raise ValueError(f"Unexpected shape of images[{i}]: {images[i].shape}")

            adj, features = mri_to_graph(mri_slice, connectivity=connectivity)

            # convert to torch tensors and move to device
            features = features.to(device)
            adj = torch.tensor(adj.todense(), dtype=torch.float32).to(device)

            # forward pass through GCN
            output = model(features, adj)

            # reshape mask and compute loss
            target_mask = masks[i].view(-1)
            output = output.view(-1, output_dim)
            loss = criterion(output, target_mask)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

            # predictions and accuracy
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target_mask.cpu().numpy()).cpu().numpy() # idk either

        epoch_loss += batch_loss / images.shape[0]

    # compute epoch-level accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")


# # ------------------------------
# # 🎯 Save Model
# # ------------------------------
# model_path = os.path.join(os.getcwd(), "results", "gcn_model.pth")
# torch.save(model.state_dict(), model_path)
# print(f"Model saved at {model_path}")
