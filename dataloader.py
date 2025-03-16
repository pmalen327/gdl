import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
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
        image_data = torch.tensor(image_data, dtype=torch.float32)

        if image_data.ndimension() == 2:
            image_data = image_data.unsqueeze(0)
        elif image_data.ndimension() == 3:
            image_data = image_data.permute(2, 0, 1)
        

        # check if mask_data has the correct shape then send to tensor
        mask_data = torch.tensor(mask_data, dtype=torch.float32)
        mask_data = mask_data[0]  # remove the batch dimension, resulting in shape (240, 240, 3)
        mask_data = torch.tensor(mask_data, dtype=torch.long)

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

for batch in dataloader:
    images = batch['image']
    masks = batch['mask']
    # will address shape mismatch in model
    print(f'Batch image shape: {images.shape}')
    print(f'Batch mask shape: {masks.shape}')
    break