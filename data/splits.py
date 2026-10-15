"""
Dataset splitting and label management for CAMELYON16.
"""
import os
import csv
import logging
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold

logger = logging.getLogger(__name__)

def load_slide_labels(label_csv_path: str) -> Dict[str, int]:
    """
    Reads a CSV with columns [slide_id, label] and returns a dict mapping slide_id to label.
    
    Args:
        label_csv_path: Path to the label CSV file.
        
    Returns:
        Dictionary mapping slide_id to integer label.
    """
    labels = {}
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"Label CSV not found: {label_csv_path}")
        
    with open(label_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['slide_id']] = int(row['label'])
    return labels

def get_camelyon16_splits(processed_dir: str, label_csv_path: str, val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Returns train, validation, and test splits for CAMELYON16 slides.
    
    Args:
        processed_dir: Directory containing processed slide H5 files.
        label_csv_path: Path to the label CSV.
        val_ratio: Proportion of training slides to use for validation.
        seed: Random seed for stratified splitting.
        
    Returns:
        Tuple of (train_slides, val_slides, test_slides) where each is a list of dictionaries.
    """
    labels_dict = load_slide_labels(label_csv_path)
    
    train_val_list = []
    test_list = []
    
    for slide_id, label in labels_dict.items():
        h5_path = os.path.join(processed_dir, f"{slide_id}_tiles.h5")
        if not os.path.exists(h5_path):
            h5_path = os.path.join(processed_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            logger.warning(f"H5 file not found for slide {slide_id} at {h5_path}. Skipping.")
            continue
            
        slide_info = {'slide_id': slide_id, 'h5_path': h5_path, 'label': label}
        
        if slide_id.startswith('test_'):
            test_list.append(slide_info)
        else:
            train_val_list.append(slide_info)
            
    if not train_val_list:
        raise ValueError("No training slides found.")
        
    # Stratified split for train/val
    slide_ids = [s['slide_id'] for s in train_val_list]
    slide_labels = [s['label'] for s in train_val_list]
    
    train_ids, val_ids = train_test_split(
        slide_ids, test_size=val_ratio, random_state=seed, stratify=slide_labels
    )
    
    train_list = [s for s in train_val_list if s['slide_id'] in train_ids]
    val_list = [s for s in train_val_list if s['slide_id'] in val_ids]
    
    return train_list, val_list, test_list

def get_camelyon16_cv_splits(processed_dir: str, label_csv_path: str = None, n_splits: int = 5, seed: int = 42) -> List[Dict[str, List[Dict]]]:
    """
    Returns cross-validation splits for CAMELYON16 slides.
    Discovers all processed H5 tile files directly from processed_dir.
    """
    labels_dict = {}
    if label_csv_path and os.path.exists(label_csv_path):
        try:
            labels_dict = load_slide_labels(label_csv_path)
        except Exception:
            pass

    # Find all tile H5 files in processed_dir
    h5_files = [f for f in os.listdir(processed_dir) if f.endswith('_tiles.h5') or f.endswith('.h5')]
    h5_files = [f for f in h5_files if not f.endswith('_features.h5') and not f.endswith('_student_feats.h5')]

    train_val_list = []
    test_list = []

    for f in sorted(h5_files):
        slide_id = f.replace('_tiles.h5', '').replace('.h5', '')
        if slide_id in labels_dict:
            label = labels_dict[slide_id]
        else:
            label = 1 if 'tumor' in slide_id.lower() else 0

        h5_path = os.path.join(processed_dir, f)
        slide_info = {'slide_id': slide_id, 'h5_path': h5_path, 'label': label}

        if slide_id.startswith('test_'):
            test_list.append(slide_info)
        else:
            train_val_list.append(slide_info)

    if not train_val_list:
        raise ValueError("No training slides found in processed_dir.")

    slide_ids = [s['slide_id'] for s in train_val_list]
    slide_labels = [s['label'] for s in train_val_list]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_splits = []
    for train_idx, val_idx in skf.split(slide_ids, slide_labels):
        train_list = [train_val_list[i] for i in train_idx]
        val_list = [train_val_list[i] for i in val_idx]

        cv_splits.append({
            'train': train_list,
            'val': val_list,
            'test': test_list
        })

    return cv_splits

def create_default_label_csv(raw_dir: str, output_csv: str) -> None:
    """
    Generates a label CSV by inferring labels from CAMELYON16 naming convention.
    
    Args:
        raw_dir: Directory containing raw CAMELYON16 slide files.
        output_csv: Path to save the output CSV.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        
    slide_files = [f for f in os.listdir(raw_dir) if f.endswith(('.tif', '.ndpi', '.svs'))]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['slide_id', 'label'])
        
        for file in slide_files:
            slide_id = os.path.splitext(file)[0]
            label = 0
            if slide_id.startswith('tumor_') or (slide_id.startswith('test_') and 'tumor' in slide_id.lower()):
                label = 1
            writer.writerow([slide_id, label])
    
    logger.info(f"Created label CSV at {output_csv} with {len(slide_files)} entries.")
