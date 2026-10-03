import os
import yaml
import torch
import numpy as np
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.aggregator import AttentionAggregator
from optimisation.trt_inference import TRTWrapper

def main():
    print("=== FastPath Speed Benchmark ===")
    
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    tile_size = config["dataset"]["tile_size"]
    trt_engine_path = "checkpoints/trt/student_fp16.engine"
    
    if not os.path.exists(trt_engine_path):
        print(f"Error: TensorRT engine not found at {trt_engine_path}.")
        return
        
    print("Loading models...")
    trt_student = TRTWrapper(trt_engine_path)
    
    device = torch.device("cuda")
    aggregator = AttentionAggregator(in_dim=128, hidden_dim=64, num_classes=1).to(device)
    aggregator.eval()
    
    # --- Benchmark 1: Tile-level Inference (ms / tile) ---
    batch_size = 512
    dummy_input = np.random.randn(batch_size, 3, tile_size, tile_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        trt_student.infer(dummy_input)
        
    num_iterations = 100
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        trt_student.infer(dummy_input)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000
    ms_per_batch = total_time_ms / num_iterations
    ms_per_tile = ms_per_batch / batch_size
    
    print(f"\nTile-level Student Inference (Batch Size = {batch_size}):")
    print(f"-> {ms_per_batch:.2f} ms per batch")
    print(f"-> {ms_per_tile:.4f} ms per tile")
    
    # --- Benchmark 2: Slide-level End-to-End ---
    # A typical CAMELYON16 slide yields ~10,000 tissue tiles at 20x.
    num_tiles_per_slide = 10000
    
    slide_features_dummy = torch.randn(num_tiles_per_slide, 128, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    simulated_trt_time_ms = ms_per_tile * num_tiles_per_slide
    
    # Actual Aggregator inference
    with torch.no_grad():
        logits, _ = aggregator(slide_features_dummy)
        
    torch.cuda.synchronize()
    aggregator_time_ms = (time.time() - start_time) * 1000
    
    total_slide_ms = simulated_trt_time_ms + aggregator_time_ms
    
    print(f"\nSlide-level Model Inference (assuming {num_tiles_per_slide} tiles/slide):")
    print(f"-> TRT Backbone: {simulated_trt_time_ms:.2f} ms")
    print(f"-> Aggregator:   {aggregator_time_ms:.2f} ms")
    print(f"-> Total Time:   {total_slide_ms:.2f} ms per slide")
    
    # --- Benchmark 3: Throughput (Slides / Hour) ---
    slides_per_second = 1000.0 / total_slide_ms
    slides_per_hour = slides_per_second * 3600
    
    print(f"\nThroughput Projection on B200 (Inference Math Only):")
    print(f"-> {slides_per_hour:.0f} slides / hour")
    
    print("\n=== Baseline Comparison ===")
    print(f"CLAM (Lu 2021)      : ~60 slides/hr")
    print(f"HDMIL (Dong 2025)   : ~84 slides/hr")
    print(f"FastPath (Ours)     : ~{slides_per_hour:.0f} slides/hr")
    
if __name__ == "__main__":
    main()
