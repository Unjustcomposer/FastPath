"""
Production-grade training script for the FastPath student model.
Uses knowledge distillation with feature-hint loss, EMA, cosine warmup,
early stopping, gradient clipping, BF16 mixed precision, DDP, and CV.
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.student import StudentModel
from models.distillation import KDLoss
from data.dataset import TileDataset, h5_worker_init_fn
from data.splits import get_camelyon16_cv_splits
from data.augmentation import get_train_transforms, get_eval_transforms
from training.utils import set_seed, EMA, CosineWarmupScheduler, EarlyStopping, MetricTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FastPath-StudentTrainer")


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

    # ── Load configs ─────────────────────────────────────────────────────
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

    # ── Data splits (CV) ─────────────────────────────────────────────────
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

    train_tile_paths = [s["h5_path"] for s in train_slides]
    train_feat_paths = [s["h5_path"].replace("_tiles.h5", "_features.h5") for s in train_slides]
    val_tile_paths = [s["h5_path"] for s in val_slides]
    val_feat_paths = [s["h5_path"].replace("_tiles.h5", "_features.h5") for s in val_slides]

    # ── Augmentation ─────────────────────────────────────────────────────
    aug_cfg = train_cfg.get("augmentation", {})
    train_transform = get_train_transforms(aug_cfg)
    val_transform = get_eval_transforms()

    # ── Datasets & Loaders ───────────────────────────────────────────────
    train_dataset = TileDataset(
        h5_paths=train_tile_paths,
        teacher_feat_paths=train_feat_paths,
        transform=train_transform,
        return_hints=True
    )
    val_dataset = TileDataset(
        h5_paths=val_tile_paths,
        teacher_feat_paths=val_feat_paths,
        transform=val_transform,
        return_hints=False
    )

    t_cfg = train_cfg.get("training", {})
    batch_size = t_cfg.get("tile_batch_size", 256)
    num_workers = t_cfg.get("num_workers", 8)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=h5_worker_init_fn,
        drop_last=True,
        sampler=train_sampler
    )
    
    # We only validate on rank 0 for simplicity
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=h5_worker_init_fn
    ) if rank == 0 else None

    # ── Model, Loss, Optimizer ───────────────────────────────────────────
    s_cfg = train_cfg.get("student", {})
    d_cfg = train_cfg.get("distillation", {})
    teacher_dim = train_cfg.get("teacher", {}).get("feature_dim", 2048)

    model = StudentModel(
        num_classes=1,
        pretrained=s_cfg.get("pretrained", True),
        feature_dim=s_cfg.get("feature_dim", 128),
        teacher_dim=teacher_dim
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    criterion = KDLoss(
        alpha=d_cfg.get("alpha", 0.3),
        beta=d_cfg.get("beta", 0.1),
        temperature=d_cfg.get("temperature", 4.0)
    ).to(device)

    # Scale LR by world_size
    lr = float(t_cfg.get("tile_lr", 1e-3)) * world_size
    min_lr = float(t_cfg.get("tile_min_lr", 1e-6))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = t_cfg.get("tile_epochs", 30)
    warmup_epochs = t_cfg.get("warmup_epochs", 3)
    patience = t_cfg.get("early_stopping_patience", 7)
    max_norm = t_cfg.get("gradient_clip_norm", 1.0)
    ema_decay = t_cfg.get("ema_decay", 0.999)

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr=min_lr,
        steps_per_epoch=len(train_loader)
    )
    
    # Track EMA and metrics on rank 0 only
    if rank == 0:
        ema = EMA(model.module if is_ddp else model, decay=ema_decay)
        early_stopping = EarlyStopping(patience=patience, mode="max")
        val_tracker = MetricTracker()
        
    train_tracker = MetricTracker()
    scaler = torch.amp.GradScaler("cuda")

    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        best_path = f"checkpoints/student_best_fold{args.fold}.pt"
        ema_path = f"checkpoints/student_ema_best_fold{args.fold}.pt"
        logger.info(f"Training up to {epochs} epochs | batch_size={batch_size} | lr={lr}")

    # ── Training Loop ────────────────────────────────────────────────────
    for epoch in range(epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_tracker.reset()

        for imgs, t_logits, pseudo_labels, t_feats in train_loader:
            imgs = imgs.to(device)
            t_logits = t_logits.to(device)
            pseudo_labels = pseudo_labels.to(device)
            t_feats = t_feats.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_logits, _, s_hints = model(imgs)
                loss = criterion(s_logits, t_logits, pseudo_labels, student_hints=s_hints, teacher_feats=t_feats)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            train_tracker.update(loss.detach().item(), imgs.size(0))
            
            if rank == 0:
                ema.update(model.module if is_ddp else model)

        # ── Validation (Rank 0 only) ─────────────────────────────────────
        stop_training = torch.tensor(0).to(device)
        
        if rank == 0:
            model_to_eval = model.module if is_ddp else model
            model_to_eval.eval()
            val_tracker.reset()
            all_preds = []
            all_labels = []

            ema.apply(model_to_eval)
            with torch.no_grad():
                for batch in val_loader:
                    imgs, t_logits, pseudo_labels = batch[0], batch[1], batch[2]
                    imgs = imgs.to(device)
                    pseudo_labels = pseudo_labels.to(device)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        s_logits, _ = model_to_eval(imgs)
                        val_loss = nn.BCEWithLogitsLoss()(s_logits, pseudo_labels.float().unsqueeze(-1) if pseudo_labels.dim() == 1 else pseudo_labels.float())

                    val_tracker.update(val_loss.item(), imgs.size(0))
                    all_preds.extend(torch.sigmoid(s_logits).float().cpu().numpy().flatten())
                    all_labels.extend(pseudo_labels.float().cpu().numpy().flatten())

            ema.restore(model_to_eval)

            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                val_auc = 0.0

            train_loss = train_tracker.compute()
            val_loss = val_tracker.compute()
            current_lr = scheduler.get_lr()

            logger.info(
                f"Epoch {epoch + 1:3d}/{epochs}  |  LR: {current_lr:.2e}  |  "
                f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val AUC: {val_auc:.4f}"
            )

            if early_stopping(val_auc, model_to_eval, best_path):
                logger.info(f"Early stopping triggered. Best AUC: {early_stopping.best_score:.4f}")
                ema.apply(model_to_eval)
                torch.save(model_to_eval.state_dict(), ema_path)
                ema.restore(model_to_eval)
                stop_training += 1
            elif early_stopping.counter == 0:
                ema.apply(model_to_eval)
                torch.save(model_to_eval.state_dict(), ema_path)
                ema.restore(model_to_eval)
                
        if is_ddp:
            dist.broadcast(stop_training, src=0)
            
        if stop_training.item() == 1:
            break

    if rank == 0:
        ema.apply(model_to_eval)
        torch.save(model_to_eval.state_dict(), f"checkpoints/student_final_fold{args.fold}.pt")
        logger.info(f"Training complete. Final EMA model saved.")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
