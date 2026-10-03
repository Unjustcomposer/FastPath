import cv2
import numpy as np
from tqdm import tqdm
from data.slide_io import SlideReader

def get_tissue_mask(thumbnail):
    """
    Generate tissue mask using Otsu thresholding on HSV image.
    """
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
    # Saturation channel is usually good for tissue detection
    s = hsv[:, :, 1]
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0

def generate_tiles(slide_path, target_mag=20, tile_size=256, min_tissue_ratio=0.5):
    """
    Yields valid tile coordinates and tissue ratios.
    """
    reader = SlideReader(slide_path)
    try:
        level = reader.get_level_for_mag(target_mag)
        downsample = reader.slide.level_downsamples[level]
        
        # Get thumbnail for mask (use lowest res)
        thumb_level = reader.slide.level_count - 1
        thumbnail = reader.get_thumbnail(thumb_level)
        thumb_downsample = reader.slide.level_downsamples[thumb_level]
        mask = get_tissue_mask(thumbnail)
        
        dims = reader.slide.level_dimensions[level]
        w_tiles = dims[0] // tile_size
        h_tiles = dims[1] // tile_size
        
        valid_tiles = []
        
        # Map tile size at `level` to thumbnail level to check mask
        scale_factor = thumb_downsample / downsample
        mask_tile_size = int(tile_size / scale_factor)
        mask_tile_size = max(1, mask_tile_size)
        
        for y in range(h_tiles):
            for x in range(w_tiles):
                # Calculate mask coordinates
                mx = int((x * tile_size) / scale_factor)
                my = int((y * tile_size) / scale_factor)
                
                mask_crop = mask[my:my+mask_tile_size, mx:mx+mask_tile_size]
                if mask_crop.size == 0:
                    continue
                    
                tissue_ratio = np.mean(mask_crop)
                if tissue_ratio >= min_tissue_ratio:
                    # location in level 0
                    loc_x = int(x * tile_size * downsample)
                    loc_y = int(y * tile_size * downsample)
                    valid_tiles.append({'x': loc_x, 'y': loc_y, 'level': level, 'size': tile_size, 'tissue_ratio': tissue_ratio})
                    
        return valid_tiles
    finally:
        reader.close()
