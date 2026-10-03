import os
import h5py
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

# Add root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.teacher import TeacherModel

class H5TileDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5f = h5py.File(h5_path, 'r')
        self.images = self.h5f['images']
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # image is (H, W, 3) uint8
        img = self.images[idx]
        # convert to tensor, normalize (CHW, float32)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img
        
    def __del__(self):
        try:
            self.h5f.close()
        except:
            pass

def main():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config["dataset"]["processed_dir"]
    if not os.path.exists(processed_dir):
        print(f"Directory {processed_dir} does not exist. Run preprocessing first.")
        return
        
    h5_files = [f for f in os.listdir(processed_dir) if f.endswith("_tiles.h5")]
    if not h5_files:
        print(f"No HDF5 files found in {processed_dir}.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize teacher model
    print("Loading teacher model...")
    # NOTE: Replace 'resnet50' with the specific backbone of your pre-trained teacher
    # and load its custom pathology weights if using CTransPath or UNI.
    model = TeacherModel(model_name='resnet50', pretrained=True).to(device)
    model.eval()
    
    # Use a large batch size since B200 has 192GB of VRAM
    batch_size = 512 
    
    for h5_file in tqdm(h5_files, desc="Extracting features"):
        slide_id = h5_file.replace("_tiles.h5", "")
        h5_path = os.path.join(processed_dir, h5_file)
        
        feat_path = os.path.join(processed_dir, f"{slide_id}_features.h5")
        
        if os.path.exists(feat_path):
            continue
            
        dataset = H5TileDataset(h5_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_features = []
        all_logits = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits, features = model(batch)
                
                all_features.append(features.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                
        if len(all_features) == 0:
            continue
            
        all_features = np.concatenate(all_features, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        
        # Save to a new feature HDF5
        with h5py.File(feat_path, 'w') as f_out:
            f_out.create_dataset("features", data=all_features, compression="lzf")
            f_out.create_dataset("logits", data=all_logits, compression="lzf")
            
            # Copy coordinates for aggregation later
            with h5py.File(h5_path, 'r') as f_in:
                if 'coords' in f_in:
                    f_out.create_dataset("coords", data=f_in['coords'][:])

if __name__ == "__main__":
    main()
