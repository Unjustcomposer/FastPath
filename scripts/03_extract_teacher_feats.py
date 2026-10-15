"""
Teacher feature extraction with FP16 mixed precision and multi-GPU support.

Usage:
    # Single GPU:
    python scripts/03_extract_teacher_feats.py

    # Multi-GPU parallel (run each in a separate terminal):
    CUDA_VISIBLE_DEVICES=0 python scripts/03_extract_teacher_feats.py --gpu_id 0 --num_gpus 4
    CUDA_VISIBLE_DEVICES=1 python scripts/03_extract_teacher_feats.py --gpu_id 1 --num_gpus 4
"""
import os, h5py, yaml, torch, numpy as np, argparse, sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.teacher import TeacherModel

class H5TileDataset(Dataset):
    def __init__(self, h5_path):
        self.h5f = h5py.File(h5_path, 'r')
        self.images = self.h5f['images']
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img - mean) / std
    def __del__(self):
        try: self.h5f.close()
        except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    with open("configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/train.yaml") as f:
        train_cfg = yaml.safe_load(f)

    processed_dir = data_cfg["dataset"]["processed_dir"]
    if not os.path.exists(processed_dir):
        print(f"{processed_dir} does not exist. Run preprocessing first.")
        return

    h5_files = sorted([f for f in os.listdir(processed_dir) if f.endswith("_tiles.h5")])
    if not h5_files:
        print(f"No HDF5 files found in {processed_dir}.")
        return

    if args.num_gpus > 1:
        h5_files = [f for i, f in enumerate(h5_files) if i % args.num_gpus == args.gpu_id]
        print(f"[GPU {args.gpu_id}/{args.num_gpus}] Processing {len(h5_files)} slides")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = train_cfg["teacher"]["backbone"]
    print(f"Loading teacher model: {backbone}...")
    model = TeacherModel(model_name=backbone).to(device)
    model.eval()

    use_fp16 = (not args.no_fp16) and device.type == "cuda"
    if use_fp16:
        print("Mixed Precision: FP16 enabled (2x speedup on Tensor Cores)")

    for h5_file in tqdm(h5_files, desc="Extracting features"):
        slide_id = h5_file.replace("_tiles.h5", "")
        h5_path = os.path.join(processed_dir, h5_file)
        feat_path = os.path.join(processed_dir, f"{slide_id}_features.h5")

        if os.path.exists(feat_path):
            continue

        dataset = H5TileDataset(h5_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

        all_features, all_logits = [], []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                logits, features = model(batch)
                all_features.append(features.float().cpu().numpy())
                all_logits.append(logits.float().cpu().numpy())

        if not all_features:
            continue

        all_features = np.concatenate(all_features, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)

        with h5py.File(feat_path, "w") as f_out:
            f_out.create_dataset("features", data=all_features, compression="lzf")
            f_out.create_dataset("logits", data=all_logits, compression="lzf")
            with h5py.File(h5_path, "r") as f_in:
                if "coords" in f_in:
                    f_out.create_dataset("coords", data=f_in["coords"][:])

    print("Feature extraction complete!")

if __name__ == "__main__":
    main()
