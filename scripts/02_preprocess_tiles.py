import os
import yaml
import h5py
import numpy as np
from tqdm import tqdm
import sys

# Add root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.slide_io import SlideReader
from data.tiling import generate_tiles

def main():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = config["dataset"]["raw_dir"]
    out_dir = config["dataset"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"Created {raw_dir}. Please place .tif slide files there.")
        return
        
    slide_files = [f for f in os.listdir(raw_dir) if f.endswith(".tif")]
    
    if not slide_files:
        print(f"No .tif slide files found in {raw_dir}.")
        return
        
    for slide_name in tqdm(slide_files, desc="Processing slides"):
        slide_path = os.path.join(raw_dir, slide_name)
        slide_id = os.path.splitext(slide_name)[0]
        
        # 1. Generate valid tile coordinates
        tiles = generate_tiles(
            slide_path, 
            target_mag=config["dataset"]["target_mag"],
            tile_size=config["dataset"]["tile_size"],
            min_tissue_ratio=config["tiling"]["min_tissue_ratio"]
        )
        
        if not tiles:
            print(f"No valid tissue found in {slide_name}")
            continue
            
        # 2. Extract and save tiles to HDF5
        h5_path = os.path.join(out_dir, f"{slide_id}_tiles.h5")
        if os.path.exists(h5_path):
            continue
        reader = SlideReader(slide_path)
        try:
            with h5py.File(h5_path, "w") as h5f:
                coords = np.array([[t['x'], t['y']] for t in tiles])
                h5f.create_dataset("coords", data=coords)
                h5f.attrs["level"] = tiles[0]["level"]
                h5f.attrs["tile_size"] = config["dataset"]["tile_size"]
                
                # Save extracted tiles
                img_shape = (len(tiles), config["dataset"]["tile_size"], config["dataset"]["tile_size"], 3)
                img_dset = h5f.create_dataset(
                    "images", 
                    shape=img_shape,
                    dtype=np.uint8,
                    chunks=(1, config["dataset"]["tile_size"], config["dataset"]["tile_size"], 3),
                    compression="lzf"
                )
                
                for i, t in enumerate(tiles):
                    img = reader.read_region((t['x'], t['y']), t['level'], (t['size'], t['size']))
                    img_dset[i] = img
        except Exception as e:
            print(f"Error processing {slide_name}: {e}")
        finally:
            reader.close()

if __name__ == "__main__":
    main()
