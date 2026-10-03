# FastPath 🔬⚡

**Fast, Accurate Breast Cancer Metastasis Detection from Whole-Slide Images using Knowledge Distillation and Hardware-Aware Optimisation**

---

## Overview

FastPath is a lightweight deep-learning pipeline for binary classification of axillary lymph-node metastasis from histopathology whole-slide images (WSIs). It combines:

- **Knowledge Distillation** — A large teacher model (ResNet-50 / CTransPath / UNI) transfers its learned representations to a compact EfficientNet-B0 student via soft-target and feature-hint losses.
- **Gated Attention Aggregation** — A lightweight CLAM-style gated attention module pools thousands of tile embeddings into a single slide-level diagnosis.
- **Hardware-Aware Deployment** — The student model is exported to ONNX and compiled into a TensorRT engine with FP16/FP8 quantisation for maximum throughput on NVIDIA B200 GPUs.

Evaluated on the **CAMELYON16** benchmark dataset with pathologist-provided ground-truth labels.

## Project Structure

```
fastpath/
├── configs/              # YAML experiment configurations
│   ├── data.yaml         # Dataset paths, tile size, magnification
│   └── train.yaml        # All training hyperparameters
├── data/                 # Data pipeline
│   ├── augmentation.py   # Medical imaging augmentations (torchvision v2)
│   ├── dataset.py        # TileDataset + SlideFeatureDataset (HDF5-backed)
│   ├── slide_io.py       # OpenSlide WSI reader
│   ├── splits.py         # CAMELYON16 label loading + stratified splits
│   ├── stain_norm.py     # Macenko stain normalisation
│   └── tiling.py         # Tissue detection + tessellation
├── models/               # Model architectures
│   ├── aggregator.py     # Gated Attention slide-level aggregator
│   ├── distillation.py   # KD loss (soft-target + feature-hint + hard-label)
│   ├── student.py        # EfficientNet-B0 student with hint projection
│   └── teacher.py        # Teacher backbone wrapper (timm)
├── training/             # Training infrastructure
│   └── utils.py          # EMA, CosineWarmupScheduler, EarlyStopping
├── optimisation/         # TensorRT deployment
│   ├── trt_builder.py    # ONNX → TensorRT engine compilation
│   └── trt_inference.py  # TRT inference wrapper with CUDA streams
├── evaluation/           # Metrics & interpretability
│   ├── calibration.py    # Temperature scaling (Guo et al., 2017)
│   ├── compare_baselines.py  # Markdown + scatter plot vs CLAM/TransMIL/HDMIL
│   ├── heatmap.py        # Attention heatmap overlays
│   └── metrics.py        # AUC, F1, sensitivity, specificity
├── scripts/              # Executable pipeline steps
│   ├── 02_preprocess_tiles.py
│   ├── 03_extract_teacher_feats.py
│   ├── 04_train_student.py
│   ├── 05_train_aggregator.py
│   ├── 06_quantize_export.py
│   ├── 07_evaluate.py
│   ├── 08_benchmark.py
│   └── 09_infer_slide.py    # ← Clinical inference entrypoint
├── docker/
│   └── Dockerfile        # Multi-stage CUDA 12.4 production image
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place CAMELYON16 `.tif` slides in `data_raw/`, then:

```bash
# Generate labels CSV from filenames
python -c "from data.splits import create_default_label_csv; create_default_label_csv('data_raw', 'data_processed/labels.csv')"

# Tile the slides and save to HDF5
python scripts/02_preprocess_tiles.py
```

### 3. Train

```bash
# Extract teacher features (one-time)
python scripts/03_extract_teacher_feats.py

# Train student via knowledge distillation
python scripts/04_train_student.py

# Train slide-level aggregator
python scripts/05_train_aggregator.py
```

### 4. Deploy

```bash
# Export to ONNX + compile TensorRT engine
python scripts/06_quantize_export.py
```

### 5. Run Inference on a Single Slide

```bash
python scripts/09_infer_slide.py \
  --slide_path /path/to/slide.tif \
  --output_dir ./results \
  --use_trt \
  --generate_heatmap
```

Output:
```
==========================================================
            F A S T P A T H   C L I N I C A L   R E P O R T
==========================================================
  Slide ID       : tumor_042
  Prediction     : Positive (Metastasis Detected)
  Confidence     : 0.9731
  Tiles Processed: 12847
  Inference Time : 3.21 s
  Speed          : 0.0312 ms/tile
  Backend        : TensorRT
==========================================================
```

### 6. Evaluate & Benchmark

```bash
python scripts/07_evaluate.py    # Accuracy metrics on test set
python scripts/08_benchmark.py   # Speed benchmarks on B200
```

## Docker

```bash
docker build -t fastpath -f docker/Dockerfile .

docker run --gpus all fastpath \
  --slide_path /data/slide.tif \
  --output_dir /output
```

## Training Features

| Feature | Implementation |
|---|---|
| Knowledge Distillation | Soft-target KL + Feature-hint MSE + Hard-label CE |
| Exponential Moving Average | Shadow parameter tracking with apply/restore |
| Learning Rate Schedule | Linear warmup → Cosine decay |
| Mixed Precision | BF16 autocast on B200 Tensor Cores |
| Early Stopping | Patience-based on validation AUC |
| Gradient Clipping | Max-norm clipping (default 1.0) |
| Data Augmentation | Flip, rotation, colour jitter, blur, cutout |
| Stain Normalisation | Macenko SVD-based |
| Confidence Calibration | Post-hoc temperature scaling |

## Citation

If you use this work, please cite:

```bibtex
@misc{fastpath2026,
  title={FastPath: Knowledge-Distilled Fast Inference for Breast Cancer Metastasis Detection},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
