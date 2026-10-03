import os
import torch
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.student import StudentModel
from optimisation.trt_builder import build_engine

def export_to_onnx(model, onnx_path, tile_size=256, batch_size=512):
    print(f"Exporting PyTorch model to ONNX: {onnx_path}")
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(batch_size, 3, tile_size, tile_size, device="cuda")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    print("ONNX export complete.")

def main():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    tile_size = config["dataset"]["tile_size"]
    batch_size = 512 # Designed for B200 throughput
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("CUDA is required for ONNX export and TensorRT compilation.")
        return
        
    os.makedirs("checkpoints/onnx", exist_ok=True)
    os.makedirs("checkpoints/trt", exist_ok=True)
    
    onnx_path = "checkpoints/onnx/student.onnx"
    engine_path = "checkpoints/trt/student_fp16.engine"
    
    # 1. Load trained PyTorch student model
    model = StudentModel(num_classes=1, pretrained=False, feature_dim=128).to(device)
    student_ckpt = "checkpoints/student_latest.pt"
    
    if os.path.exists(student_ckpt):
        model.load_state_dict(torch.load(student_ckpt))
        print(f"Loaded student weights from {student_ckpt}")
    else:
        print("Warning: No trained weights found. Exporting randomly initialized model for testing.")
        
    # 2. Export PyTorch to ONNX
    if not os.path.exists(onnx_path):
        export_to_onnx(model, onnx_path, tile_size=tile_size, batch_size=batch_size)
    else:
        print(f"ONNX model already exists at {onnx_path}. Skipping export.")
        
    # 3. Compile ONNX into TensorRT Engine
    # We default to FP16 because it's incredibly robust and fast.
    # To use native FP8 on the B200, change `precision='fp8'` below.
    build_engine(onnx_path, engine_path, precision='fp16', batch_size=batch_size, tile_size=tile_size)

if __name__ == "__main__":
    main()
