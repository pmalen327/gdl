import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import networkx as nx
from dotenv import load_dotenv
from torchvision import transforms

# class to load and extract information from MRI data in h5 format
class MRI_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # data_dir (string): Directory with all the h5 files.
        # transform (callable, optional): Optional transform to be applied on a sample.
        self.data_dir = data_dir
        self.transform = transform
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_name = self.files[index]
        file_path = os.path.join(self.data_dir, file_name)

        with h5py.File(file_path, 'r') as f:
            image_data = f['image'][:]
            mask_data = f['mask'][:]

        # check if image_data has the correct shape then send to tensor
        
        image_data = image_data[0]  # Remove the batch dimension, resulting in shape (240, 240, 4)
        image_data = torch.from_numpy(image_data).float()

        if image_data.ndimension() == 2:
            image_data = image_data.unsqueeze(0)
        elif image_data.ndimension() == 3:
            image_data = image_data.permute(2, 0, 1)
        

        # check if mask_data has the correct shape then send to tensor
        mask_data = torch.from_numpy(mask_data).float()
        mask_data = mask_data[0]  # remove the batch dimension, resulting in shape (240, 240, 3)
        mask_data = mask_data.long()

        # apply transformations if any
        if self.transform:
            image_data = self.transform(image_data)
        
        image_data = np.squeeze(image_data)  # remove any singleton dimensions

        # return dict of image and corresponding mask
        return {'image': image_data, 'mask': mask_data}

load_dotenv()
data_dir = os.getenv('DATA_PATH') # set the data path in the .env file
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

dataset = MRI_Dataset(data_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# for batch in dataloader:
#     images = batch['image']
#     masks = batch['mask']
#     # will address shape mismatch in model
#     print(f'Batch image shape: {images.shape}')
#     print(f'Batch mask shape: {masks.shape}')
#     break

# convert MRI slice to graph representation
def mri_to_graph(mri_slice, connectivity=4):
    """
    Convert an MRI slice to a graph.

    Args:
        mri_slice (np.array): Shape (240, 240, 4), an individual MRI slice.
        connectivity (int): 4 or 8 for grid graph connectivity.

    Returns:
        adj (csr_matrix): Sparse adjacency matrix.
        features (torch.Tensor): Node features as tensor of shape [N, 4].
    """
    height, width, channels = mri_slice.shape
    G = nx.grid_2d_graph(height,width)

    # if we want 8-connectivity, add diagonal edges
    if connectivity == 8:
        G.add_edges_from([
            ((i, j), (i + di, j + dj))
            for i in range(height-1) for j in range(width)
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            if 0 <= i + di < height and 0 <= j + dj < width
        ])

    # flatten the 2D grid to 1D node list
    features = mri_slice.reshape(-1, channels)

    # create adjacency matrix as sparse matrix
    adj = nx.to_scipy_sparse_array(G, format='csr')

    # return adjacency matrix and node features as tensor
    features = torch.tensor(features, dtype=torch.float32)
    return adj, features