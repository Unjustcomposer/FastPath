"""
End-to-end clinical inference pipeline for a SINGLE raw .tif slide.
This is the deployment entrypoint for FastPath.

Usage:
    python scripts/09_infer_slide.py --slide_path /path/to/slide.tif --output_dir ./results
    python scripts/09_infer_slide.py --slide_path /path/to/slide.tif --output_dir ./results --use_trt --generate_heatmap
"""

import argparse
import json
import logging
import time
import os
import sys

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.slide_io import SlideReader
from data.tiling import generate_tiles
from data.augmentation import get_eval_transforms
from models.student import StudentModel
from models.aggregator import AttentionAggregator
from evaluation.heatmap import generate_slide_heatmap, generate_tile_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FastPath-Inference")


def load_student_pytorch(checkpoint_dir: str, device: torch.device) -> StudentModel:
    """Loads the student model from a PyTorch checkpoint."""
    model = StudentModel(num_classes=1, pretrained=False, feature_dim=128, teacher_dim=2048).to(device)
    ckpt_path = os.path.join(checkpoint_dir, "student_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "student_final.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Loaded student weights from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No student checkpoint found in {checkpoint_dir}")
    model.eval()
    return model


def load_aggregator(checkpoint_dir: str, device: torch.device) -> AttentionAggregator:
    """Loads the attention aggregator from a PyTorch checkpoint."""
    model = AttentionAggregator(in_dim=128, hidden_dim=128, num_classes=1).to(device)
    ckpt_path = os.path.join(checkpoint_dir, "aggregator_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "aggregator_latest.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Loaded aggregator weights from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No aggregator checkpoint found in {checkpoint_dir}")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="FastPath: End-to-end clinical inference for a single .tif whole-slide image."
    )
    parser.add_argument("--slide_path", type=str, required=True, help="Path to raw .tif slide")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output report and heatmap")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory containing model checkpoints")
    parser.add_argument("--use_trt", action="store_true", help="Use TensorRT engine instead of PyTorch")
    parser.add_argument("--generate_heatmap", action="store_true", help="Generate and save attention heatmap overlay")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size in pixels")
    parser.add_argument("--target_mag", type=float, default=20.0, help="Target magnification for tiling")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for tile inference")
    parser.add_argument("--tta", action="store_true", help="Use 4x Test-Time Augmentation (rotations)")
    args = parser.parse_args()

    if not os.path.exists(args.slide_path):
        logger.error(f"Slide file not found: {args.slide_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    total_start = time.time()

    # ── 1. Load slide and generate tissue tiles ──────────────────────────
    logger.info(f"Loading slide: {args.slide_path}")
    tiles_meta = generate_tiles(
        args.slide_path,
        target_mag=args.target_mag,
        tile_size=args.tile_size,
        min_tissue_ratio=0.5
    )
    num_tiles = len(tiles_meta)
    logger.info(f"Detected {num_tiles} tissue tiles")

    if num_tiles == 0:
        logger.error("No tissue tiles found in slide. Aborting.")
        return

    # ── 2. Load models ───────────────────────────────────────────────────
    if args.use_trt:
        try:
            from optimisation.trt_inference import TRTWrapper
            engine_path = os.path.join(args.checkpoint_dir, "trt", "student_fp16.engine")
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TRT engine not found at {engine_path}")
            student = TRTWrapper(engine_path)
            use_trt = True
            logger.info("Loaded TensorRT student engine")
        except Exception as e:
            logger.warning(f"TRT load failed ({e}), falling back to PyTorch")
            student = load_student_pytorch(args.checkpoint_dir, device)
            use_trt = False
    else:
        student = load_student_pytorch(args.checkpoint_dir, device)
        use_trt = False

    aggregator = load_aggregator(args.checkpoint_dir, device)

    # ── 3. Extract tile features ─────────────────────────────────────────
    logger.info("Running tile-level inference...")
    reader = SlideReader(args.slide_path)
    transform = get_eval_transforms()

    all_features = []
    inference_start = time.time()
    
    if use_trt:
        logger.info("Capturing CUDA Graph for TensorRT engine...")
        dummy_shape = (args.batch_size, 3, args.tile_size, args.tile_size)
        dummy_batch = torch.zeros(dummy_shape, dtype=torch.float32, device=device)
        student.capture_graph(dummy_batch)

    for i in range(0, num_tiles, args.batch_size):
        batch_meta = tiles_meta[i : i + args.batch_size]
        actual_batch_size = len(batch_meta)

        # Read tiles from the slide
        batch_imgs = []
        for t in batch_meta:
            img = reader.read_region((t["x"], t["y"]), t["level"], (t["size"], t["size"]))
            img_tensor = transform(img)
            batch_imgs.append(img_tensor)

        batch_tensor = torch.stack(batch_imgs).to(device)

        if use_trt and actual_batch_size < args.batch_size:
            pad_size = args.batch_size - actual_batch_size
            pad_tensor = torch.zeros((pad_size, 3, args.tile_size, args.tile_size), dtype=batch_tensor.dtype, device=device)
            batch_tensor = torch.cat([batch_tensor, pad_tensor], dim=0)

        def run_inference(tensor):
            if use_trt:
                outputs = student.infer_graph(tensor)
                return outputs[1]
            else:
                with torch.no_grad():
                    _, feats = student(tensor)
                return feats

        if args.tta:
            feats_list = []
            for k in range(4):
                rotated_tensor = torch.rot90(batch_tensor, k, [2, 3])
                feats = run_inference(rotated_tensor)
                feats_list.append(feats)
            feats = torch.stack(feats_list).mean(dim=0)
        else:
            feats = run_inference(batch_tensor)

        if use_trt and actual_batch_size < args.batch_size:
            feats = feats[:actual_batch_size]

        all_features.append(feats)

    reader.close()
    all_features = torch.cat(all_features, dim=0)  # (num_tiles, 128)
    tile_inference_ms = (time.time() - inference_start) * 1000

    # ── 4. Slide-level aggregation ───────────────────────────────────────
    logger.info("Running slide-level aggregation...")
    with torch.no_grad():
        logits, attention_weights = aggregator(all_features)  # logits: (1, 1), attn: (1, 1, N)

    raw_prob = torch.sigmoid(logits).item()

    # ── 5. Apply temperature scaling calibration (if available) ──────────
    temp_path = os.path.join(args.checkpoint_dir, "temperature.json")
    if os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            temperature = json.load(f).get("temperature", 1.0)
        calibrated_logit = logits.item() / temperature
        confidence = float(1.0 / (1.0 + np.exp(-calibrated_logit)))  # sigmoid
        logger.info(f"Applied calibration temperature: {temperature:.4f}")
    else:
        confidence = raw_prob
        logger.info("No calibration file found. Using raw sigmoid probability.")

    prediction = "Positive (Metastasis Detected)" if confidence >= 0.5 else "Negative (No Metastasis)"

    total_time = time.time() - total_start

    # ── 6. Identify top-attended tiles ───────────────────────────────────
    attn_np = attention_weights.squeeze().cpu().numpy()  # (num_tiles,)
    top_k = min(10, num_tiles)
    top_indices = np.argsort(attn_np)[::-1][:top_k]
    top_coords = [{"x": tiles_meta[int(idx)]["x"], "y": tiles_meta[int(idx)]["y"], "attention": float(attn_np[idx])} for idx in top_indices]

    # ── 7. Generate JSON report ──────────────────────────────────────────
    slide_id = os.path.splitext(os.path.basename(args.slide_path))[0]
    report = {
        "slide_id": slide_id,
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "raw_probability": round(raw_prob, 6),
        "num_tiles_processed": num_tiles,
        "tile_inference_ms": round(tile_inference_ms, 2),
        "total_inference_seconds": round(total_time, 3),
        "ms_per_tile": round(tile_inference_ms / num_tiles, 4),
        "model_backend": "TensorRT" if use_trt else "PyTorch",
        "top_attended_tiles": top_coords[:5],
    }

    report_path = os.path.join(args.output_dir, f"{slide_id}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Saved report to {report_path}")

    # ── 8. Generate heatmap (optional) ───────────────────────────────────
    if args.generate_heatmap:
        logger.info("Generating attention heatmap...")
        coords_list = [(t["x"], t["y"]) for t in tiles_meta]
        heatmap_path = os.path.join(args.output_dir, f"{slide_id}_heatmap.png")
        generate_slide_heatmap(
            slide_path=args.slide_path,
            tile_coords=coords_list,
            attention_weights=attn_np,
            tile_size=args.tile_size,
            output_path=heatmap_path,
        )

        # Also generate top-K tile grid
        logger.info("Generating top-K tile grid...")
        reader2 = SlideReader(args.slide_path)
        top_tiles_imgs = []
        top_scores = []
        for idx in top_indices:
            t = tiles_meta[int(idx)]
            img = reader2.read_region((t["x"], t["y"]), t["level"], (t["size"], t["size"]))
            top_tiles_imgs.append(img)
            top_scores.append(float(attn_np[idx]))
        reader2.close()

        grid_path = os.path.join(args.output_dir, f"{slide_id}_top_tiles.png")
        generate_tile_grid(top_tiles_imgs, top_scores, grid_path)

    # ── 9. Print clinical summary ────────────────────────────────────────
    print("\n" + "=" * 58)
    print("            F A S T P A T H   C L I N I C A L   R E P O R T")
    print("=" * 58)
    print(f"  Slide ID       : {report['slide_id']}")
    print(f"  Prediction     : {report['prediction']}")
    print(f"  Confidence     : {report['confidence']:.4f}")
    print(f"  Tiles Processed: {report['num_tiles_processed']}")
    print(f"  Inference Time : {report['total_inference_seconds']:.2f} s")
    print(f"  Speed          : {report['ms_per_tile']:.4f} ms/tile")
    print(f"  Backend        : {report['model_backend']}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
