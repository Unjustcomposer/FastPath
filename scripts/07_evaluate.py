import os
import yaml
import h5py
import torch
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.aggregator import AttentionAggregator
from optimisation.trt_inference import TRTWrapper
from evaluation.metrics import compute_classification_metrics

def get_slide_labels(slide_id):
    # Dummy label logic for CAMELYON16 (Assume tumor_* is 1, normal_* is 0)
    return 1.0 if "tumor" in slide_id.lower() else 0.0

def main():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config["dataset"]["processed_dir"]
    device = torch.device("cuda")
    
    trt_engine_path = "checkpoints/trt/student_fp16.engine"
    aggregator_ckpt = "checkpoints/aggregator_latest.pt"
    
    if not os.path.exists(trt_engine_path):
        print(f"Error: TensorRT engine not found at {trt_engine_path}. Run script 06 first.")
        return
        
    print("Loading TensorRT Engine...")
    trt_student = TRTWrapper(trt_engine_path)
    
    print("Loading Aggregator...")
    aggregator = AttentionAggregator(in_dim=128, hidden_dim=64, num_classes=1).to(device)
    if os.path.exists(aggregator_ckpt):
        aggregator.load_state_dict(torch.load(aggregator_ckpt))
    else:
        print("Warning: Aggregator weights not found. Using random initialization.")
    aggregator.eval()
    
    # Gather slides
    h5_files = [f for f in os.listdir(processed_dir) if f.endswith("_tiles.h5")]
    if not h5_files:
        print("No processed slide files found.")
        return
        
    all_probs = []
    all_targets = []
    
    print("Running End-to-End Evaluation...")
    for h5_file in tqdm(h5_files):
        slide_id = h5_file.replace("_tiles.h5", "")
        h5_path = os.path.join(processed_dir, h5_file)
        
        target = get_slide_labels(slide_id)
        all_targets.append(target)
        
        with h5py.File(h5_path, 'r') as f_in:
            images = f_in['images']
            num_tiles = len(images)
            batch_size = 512
            
            slide_features = []
            
            for i in range(0, num_tiles, batch_size):
                batch_imgs = images[i:i+batch_size]
                
                # Preprocess on CPU to array
                batch_imgs = batch_imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
                batch_imgs = (batch_imgs - mean) / std
                
                # Run TRT Inference (outputs: [logits, features])
                outputs = trt_student.infer(batch_imgs)
                feats = outputs[1] 
                
                slide_features.append(torch.from_numpy(feats).to(device))
                
            slide_features = torch.cat(slide_features, dim=0)
            
            with torch.no_grad():
                logits, _ = aggregator(slide_features)
                prob = torch.sigmoid(logits).item()
                all_probs.append(prob)
                
    metrics = compute_classification_metrics(np.array(all_targets), np.array(all_probs))
    
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
