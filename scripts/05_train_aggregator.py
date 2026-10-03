"""
Production-grade training script for the FastPath slide-level attention aggregator.
Pre-extracts student features into HDF5, then trains the aggregator with
early stopping, cosine warmup, and slide-level AUC validation. Supports DDP & CV.
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.student import StudentModel
from models.aggregator import AttentionAggregator
from data.splits import get_camelyon16_cv_splits
from data.augmentation import get_eval_transforms
from data.dataset import SlideFeatureDataset
from training.utils import set_seed, CosineWarmupScheduler, EarlyStopping, MetricTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FastPath-AggregatorTrainer")


def extract_student_features(
    student: nn.Module,
    slide_list: list,
    processed_dir: str,
    device: torch.device,
    batch_size: int = 512,
    rank: int = 0
) -> None:
    """
    Runs frozen student model over all tiles in each slide and saves
    per-slide student features to HDF5 files.
    """
    student.eval()
    transform = get_eval_transforms()

    iterator = tqdm(slide_list, desc="Extracting student features") if rank == 0 else slide_list

    for slide_info in iterator:
        slide_id = slide_info["slide_id"]
        tile_h5_path = slide_info["h5_path"]
        out_path = os.path.join(processed_dir, f"{slide_id}_student_feats.h5")

        if os.path.exists(out_path):
            continue

        if not os.path.exists(tile_h5_path):
            if rank == 0:
                logger.warning(f"Tile HDF5 not found for {slide_id}: {tile_h5_path}")
            continue

        all_feats = []
        with h5py.File(tile_h5_path, "r") as h5f:
            if "images" not in h5f:
                if rank == 0:
                    logger.warning(f"No 'images' dataset in {tile_h5_path}. Skipping.")
                continue

            images = h5f["images"]
            num_tiles = len(images)

            with torch.no_grad():
                for i in range(0, num_tiles, batch_size):
                    batch_np = images[i : i + batch_size]

                    batch_tensors = []
                    for tile in batch_np:
                        batch_tensors.append(transform(tile))
                    batch = torch.stack(batch_tensors).to(device)

                    _, feats = student(batch)
                    all_feats.append(feats.cpu().numpy())

        if all_feats:
            all_feats = np.concatenate(all_feats, axis=0)
            with h5py.File(out_path, "w") as f_out:
                f_out.create_dataset("student_features", data=all_feats, compression="lzf")
            if rank == 0:
                logger.info(f"Saved {all_feats.shape[0]} features for {slide_id}")
        else:
            if rank == 0:
                logger.warning(f"No features extracted for {slide_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help="Local rank for DDP")
    args, _ = parser.parse_known_args()

    # ── DDP Setup ────────────────────────────────────────────────────────
    is_ddp = args.local_rank != -1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        args.local_rank = 0

    with open("configs/train.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    with open("configs/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    if rank == 0:
        logger.info(f"Device: {device} | World Size: {world_size} | Fold: {args.fold}")

    processed_dir = data_cfg["dataset"]["processed_dir"]
    label_csv = os.path.join(processed_dir, "labels.csv")

    cv_splits = get_camelyon16_cv_splits(
        processed_dir=processed_dir,
        label_csv_path=label_csv,
        n_splits=5,
        seed=seed
    )
    if args.fold >= len(cv_splits):
        raise ValueError(f"Fold {args.fold} out of bounds for {len(cv_splits)} splits.")

    train_slides = cv_splits[args.fold]['train']
    val_slides = cv_splits[args.fold]['val']

    if rank == 0:
        logger.info(f"Train slides: {len(train_slides)}, Val slides: {len(val_slides)}")

    # ── Load frozen student ──────────────────────────────────────────────
    s_cfg = train_cfg.get("student", {})
    teacher_dim = train_cfg.get("teacher", {}).get("feature_dim", 2048)

    student = StudentModel(
        num_classes=1,
        pretrained=False,
        feature_dim=s_cfg.get("feature_dim", 128),
        teacher_dim=teacher_dim
    ).to(device)

    # Use fold-specific student checkpoint if available
    ckpt = f"checkpoints/student_ema_best_fold{args.fold}.pt"
    if not os.path.exists(ckpt):
        ckpt = f"checkpoints/student_best_fold{args.fold}.pt"
    
    if os.path.exists(ckpt):
        student.load_state_dict(torch.load(ckpt, map_location=device))
        if rank == 0:
            logger.info(f"Loaded student weights from {ckpt}")
    else:
        if rank == 0:
            logger.warning(f"Student checkpoint not found at {ckpt}. Using random weights.")

    # Only extract on rank 0 to avoid race conditions writing HDF5
    if rank == 0:
        all_slides = train_slides + val_slides
        extract_student_features(student, all_slides, processed_dir, device, batch_size=512, rank=rank)
    
    if is_ddp:
        dist.barrier()

    del student
    torch.cuda.empty_cache()

    for s in train_slides:
        s["h5_path"] = os.path.join(processed_dir, f"{s['slide_id']}_student_feats.h5")
    for s in val_slides:
        s["h5_path"] = os.path.join(processed_dir, f"{s['slide_id']}_student_feats.h5")

    train_dataset = SlideFeatureDataset(train_slides, processed_dir)
    val_dataset = SlideFeatureDataset(val_slides, processed_dir)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if rank == 0 else None

    # ── Aggregator model ─────────────────────────────────────────────────
    a_cfg = train_cfg.get("aggregator", {})
    t_cfg = train_cfg.get("training", {})

    model = AttentionAggregator(
        in_dim=a_cfg.get("in_dim", 128),
        hidden_dim=a_cfg.get("hidden_dim", 128),
        num_classes=1
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    criterion = nn.BCEWithLogitsLoss()
    agg_lr = float(t_cfg.get("aggregator_lr", 2e-4)) * world_size
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=agg_lr, weight_decay=weight_decay)

    epochs = t_cfg.get("aggregator_epochs", 100)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=2,
        total_epochs=epochs,
        min_lr=1e-6,
        steps_per_epoch=len(train_loader)
    )
    
    if rank == 0:
        early_stopping = EarlyStopping(patience=15, mode="max")
        os.makedirs("checkpoints", exist_ok=True)
        best_path = f"checkpoints/aggregator_best_fold{args.fold}.pt"
        logger.info(f"Training aggregator for up to {epochs} epochs | fold={args.fold}")

    train_tracker = MetricTracker()

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_tracker.reset()

        for feats, label in train_loader:
            feats = feats.to(device)
            label = label.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            logits, _ = model(feats)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_tracker.update(loss.item(), 1)

        # ── Validation (Rank 0 only) ─────────────────────────────────────
        stop_training = torch.tensor(0).to(device)

        if rank == 0:
            model_to_eval = model.module if is_ddp else model
            model_to_eval.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for feats, label in val_loader:
                    feats = feats.to(device)
                    logits, _ = model_to_eval(feats)
                    all_preds.append(torch.sigmoid(logits).item())
                    all_labels.append(label.item())

            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                val_auc = 0.0

            train_loss = train_tracker.compute()
            logger.info(f"Epoch {epoch + 1:3d}/{epochs}  |  Train Loss: {train_loss:.4f}  |  Val AUC: {val_auc:.4f}")

            if early_stopping(val_auc, model_to_eval, best_path):
                logger.info(f"Early stopping at epoch {epoch + 1}. Best AUC: {early_stopping.best_score:.4f}")
                stop_training += 1
                
        if is_ddp:
            dist.broadcast(stop_training, src=0)
            
        if stop_training.item() == 1:
            break

    if rank == 0:
        logger.info("Aggregator training complete.")
        
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
