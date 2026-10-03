"""
PyTorch datasets for FastPath model training.
"""
import h5py
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from typing import List, Dict, Callable, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def h5_worker_init_fn(worker_id: int) -> None:
    """
    Worker init function to handle HDF5 file descriptors in multiprocessing.
    """
    # H5 handles will be opened within the worker process during the first __getitem__ call 
    # to avoid pickling/threading issues.
    pass


class TileDataset(Dataset):
    """
    Lazily loads tiles from HDF5. Applies augmentation.
    """
    def __init__(self, h5_paths: List[str], teacher_feat_paths: List[str], transform: Optional[Callable] = None, return_hints: bool = False):
        """
        Args:
            h5_paths: List of paths to HDF5 files containing tiles.
            teacher_feat_paths: List of paths to HDF5 files containing teacher features/logits.
            transform: Transformation function.
            return_hints: Whether to return teacher features in addition to logits.
        """
        self.h5_paths = h5_paths
        self.teacher_feat_paths = teacher_feat_paths
        self.transform = transform
        self.return_hints = return_hints
        
        if len(self.h5_paths) != len(self.teacher_feat_paths):
            raise ValueError("Number of h5 paths and teacher feature paths must match.")
            
        self.index_map = []
        
        # Build an index map
        for file_idx, (h5_p, t_p) in enumerate(zip(self.h5_paths, self.teacher_feat_paths)):
            try:
                with h5py.File(h5_p, 'r') as f:
                    num_tiles = len(f['tiles'])
                    for tile_idx in range(num_tiles):
                        self.index_map.append((file_idx, tile_idx))
            except Exception as e:
                logger.error(f"Failed to read {h5_p}: {e}")
                
        self.file_handles = {}

    def __len__(self) -> int:
        return len(self.index_map)
        
    def _get_handles(self, file_idx: int):
        # Open handles lazily and keep them per-worker
        if file_idx not in self.file_handles:
            h5_p = self.h5_paths[file_idx]
            t_p = self.teacher_feat_paths[file_idx]
            self.file_handles[file_idx] = (h5py.File(h5_p, 'r'), h5py.File(t_p, 'r'))
        return self.file_handles[file_idx]

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        file_idx, tile_idx = self.index_map[idx]
        
        h5_f, t_f = self._get_handles(file_idx)
        
        image = h5_f['tiles'][tile_idx]
        teacher_logit = t_f['logits'][tile_idx]
        pseudo_label = np.argmax(teacher_logit)
        
        if self.transform:
            image = self.transform(image)
            
        teacher_logit_tensor = torch.from_numpy(teacher_logit).float()
        pseudo_label_tensor = torch.tensor(pseudo_label).long()
            
        if self.return_hints:
            teacher_feature = t_f['features'][tile_idx]
            teacher_feature_tensor = torch.from_numpy(teacher_feature).float()
            return image, teacher_logit_tensor, pseudo_label_tensor, teacher_feature_tensor
            
        return image, teacher_logit_tensor, pseudo_label_tensor
        
    def __del__(self):
        for h5_f, t_f in self.file_handles.values():
            try:
                h5_f.close()
                t_f.close()
            except:
                pass


class SlideFeatureDataset(Dataset):
    """
    Dataset for aggregator training. Loads pre-extracted student features from HDF5.
    """
    def __init__(self, slide_list: List[Dict], processed_dir: str):
        """
        Args:
            slide_list: List of dictionaries containing slide info.
            processed_dir: Directory containing processed H5 files with student features.
        """
        self.slide_list = slide_list
        self.processed_dir = processed_dir
        
    def __len__(self) -> int:
        return len(self.slide_list)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        slide_info = self.slide_list[idx]
        label = slide_info['label']
        h5_path = slide_info['h5_path']
        
        try:
            with h5py.File(h5_path, 'r') as f:
                features = f['student_features'][:]
        except Exception as e:
            logger.error(f"Error reading {h5_path}: {e}")
            features = np.zeros((1, 128), dtype=np.float32)
            
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(label).float()
        
        return features_tensor, label_tensor
