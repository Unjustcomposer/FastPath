import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

def generate_slide_heatmap(slide_path: str, tile_coords: List[Tuple[int, int]], 
                           attention_weights: np.ndarray, tile_size: int, 
                           output_path: str, alpha: float = 0.5) -> None:
    """
    Generates a heatmap overlay on a slide thumbnail using attention weights.
    Red corresponds to high attention (tumour), blue to low.
    
    Args:
        slide_path: Path to the slide image.
        tile_coords: List of (x, y) coordinates for the tiles.
        attention_weights: Numpy array of attention scores corresponding to coords.
        tile_size: The size of each tile in pixels.
        output_path: Where to save the heatmap PNG.
        alpha: Transparency level for the heatmap overlay.
    """
    try:
        # Assuming we can read a generic image for the thumbnail
        img = Image.open(slide_path).convert("RGBA")
        width, height = img.size
        
        heatmap = np.zeros((height, width))
        for (x, y), weight in zip(tile_coords, attention_weights):
            if x < width and y < height:
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                heatmap[y:y_end, x:x_end] = weight
                
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        cmap = plt.get_cmap('jet') # jet goes from blue (low) to red (high)
        ax.imshow(heatmap, cmap=cmap, alpha=alpha)
        ax.axis('off')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        logger.info(f"Saved slide heatmap to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate slide heatmap: {e}")

def generate_tile_grid(top_k_tiles: List[np.ndarray], attention_scores: List[float], 
                       output_path: str, cols: int = 5) -> None:
    """
    Shows the top-K most attended tiles in a grid with attention scores as titles.
    
    Args:
        top_k_tiles: List of image tiles (numpy arrays or PIL Images).
        attention_scores: Corresponding attention scores for the tiles.
        output_path: Where to save the tile grid PNG.
        cols: Number of columns in the grid.
    """
    num_tiles = len(top_k_tiles)
    if num_tiles == 0:
        logger.warning("No tiles provided for grid generation.")
        return
        
    rows = int(np.ceil(num_tiles / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Handle single row/col cases
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < num_tiles:
            axes[i].imshow(top_k_tiles[i])
            axes[i].set_title(f"Attn: {attention_scores[i]:.4f}")
            axes[i].axis('off')
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved tile grid to {output_path}")
