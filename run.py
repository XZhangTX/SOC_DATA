import math
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# -------- Dataset --------
class SpectrumSOCDataset(Dataset):
    def __init__(self, csv_path: str, amp_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Dataset for combined CSVs with per-frequency triplets (amp, sin, cos).

        - Only normalize amplitude channel across samples; sin/cos remain unchanged.
        - If amp_stats provided (mean, std), apply them; else compute from this dataset.
        """
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        assert cols[-1].lower() == "soc", "最后一列必须是 soc"

        X_flat = df[cols[:-1]].astype(np.float32).values  # [N, 3*L]
        y = df[cols[-1]].astype(np.float32).values        # [N]

        assert X_flat.shape[1] % 3 == 0, "特征维应为3的倍数(amp,sin,cos)"
        n_freq = X_flat.shape[1] // 3
        X = X_flat.reshape(len(df), n_freq, 3)

        # normalize amp only
        amp = X[:, :, 0]
        if amp_stats is None:
            mean = amp.mean(axis=0, keepdims=True)
            std = amp.std(axis=0, keepdims=True) + 1e-6
        else:
            mean, std = amp_stats
        X[:, :, 0] = (amp - mean) / std

        y = y / 100.0

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.n_freq = n_freq
        self.amp_mean = mean.astype(np.float32)
        self.amp_std = std.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]                # [n_freq, 3]
        y = self.y[idx]                # []
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# -------- Positional Encoding (sinusoidal for frequency index) --------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [L, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


# -------- iTransformer-like encoder: variables-as-tokens --------
class ITransformerEncoder(nn.Module):
    def __init__(
        self,
        d_in: int = 3,          # per-token feature dim: [amp, sin, cos]
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        n_freq: int = 512
    ):
        super().__init__()
        self.n_freq = n_freq

        # token embedding
        self.proj = nn.Linear(d_in, d_model)

        # learnable frequency index embedding
        self.freq_embed = nn.Embedding(n_freq, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_freq + 1)

        # Frequency Attention: content-based scalar per token
        self.freq_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        self.last_freq_attn: Optional[torch.Tensor] = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, 3]  (L = n_freq)
        return: [B]
        """
        B, L, _ = x.size()
        assert L == self.n_freq, f"输入频点数{L} 与模型配置{self.n_freq} 不一致"

        h = self.proj(x)  # [B, L, d_model]

        # frequency index embedding
        idx = torch.arange(L, device=x.device)
        h = h + self.freq_embed(idx)[None, :, :]

        # Frequency attention before encoder
        attn_logits = self.freq_attn(h).squeeze(-1)   # [B, L]
        attn = torch.softmax(attn_logits, dim=1)      # [B, L]
        self.last_freq_attn = attn.detach().cpu()
        h = h * attn.unsqueeze(-1)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,d_model]
        h = torch.cat([cls, h], dim=1)          # [B, 1+L, d_model]

        # positional encoding (include CLS)
        h = self.pos_enc(h)

        h = self.encoder(h)                     # [B, 1+L, d_model]
        h_cls = self.norm(h[:, 0, :])           # [B, d_model]
        y = self.head(h_cls).squeeze(-1)        # [B]
        return y


# -------- 训练/验证 --------
def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip: Optional[float] = 1.0):
    model.train()
    losses = []
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, collect_attn: bool = False, collect_preds: bool = False):
    model.eval()
    ys, ps = [], []
    attn_list = [] if collect_attn else None
    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
        if collect_attn and getattr(model, 'last_freq_attn', None) is not None:
            # last_freq_attn: [B, L] on CPU
            attn_list.append(model.last_freq_attn.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p, squared=False)
    have_attn = collect_attn and attn_list
    attn_avg = np.concatenate(attn_list, axis=0).mean(axis=0) if have_attn else None
    if collect_preds and have_attn:
        return mae, rmse, attn_avg, y, p
    if collect_preds:
        return mae, rmse, y, p
    if have_attn:
        return mae, rmse, attn_avg
    return mae, rmse


def main(args):
    # Prefer combined_*.csv and split by cycle (group-aware). Fallbacks supported.
    comb_dir = os.path.join("data", "combined")
    files = sorted(Path(comb_dir).glob("combined_*.csv"))
    if files:
        frames = []
        groups = []
        for f in files:
            df = pd.read_csv(f)
            frames.append(df)
            groups.extend([f.stem] * len(df))
        all_df = pd.concat(frames, ignore_index=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=42)
        train_idx, val_idx = next(gss.split(all_df, groups=groups))
        df_train = all_df.iloc[train_idx].reset_index(drop=True)
        df_val = all_df.iloc[val_idx].reset_index(drop=True)
        os.makedirs(comb_dir, exist_ok=True)
        df_train.to_csv(os.path.join(comb_dir, "train.csv"), index=False)
        df_val.to_csv(os.path.join(comb_dir, "val.csv"), index=False)
        # Datasets with shared amp normalization
        df_train.to_csv("data/_split_train.csv", index=False)
        df_val.to_csv("data/_split_val.csv", index=False)
        train_ds = SpectrumSOCDataset("data/_split_train.csv")
        amp_stats = (train_ds.amp_mean, train_ds.amp_std)
        val_ds = SpectrumSOCDataset("data/_split_val.csv", amp_stats=amp_stats)
        n_freq = train_ds.n_freq
    elif os.path.exists(os.path.join(comb_dir, "val.csv")):
        train_path = os.path.join(comb_dir, "train.csv")
        val_path = os.path.join(comb_dir, "val.csv")
        train_ds = SpectrumSOCDataset(train_path)
        amp_stats = (train_ds.amp_mean, train_ds.amp_std)
        val_ds = SpectrumSOCDataset(val_path, amp_stats=amp_stats)
        assert train_ds.n_freq == val_ds.n_freq, "train/val 的频点数不一致"
        n_freq = train_ds.n_freq
    else:
        # fallback: split a single train.csv at sample level
        full_path = os.path.join(comb_dir, "train.csv")
        df = pd.read_csv(full_path)
        train_df, val_df = train_test_split(df, test_size=args.test_ratio, random_state=42)
        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/_split_train.csv", index=False)
        val_df.to_csv("data/_split_val.csv", index=False)
        train_ds = SpectrumSOCDataset("data/_split_train.csv")
        amp_stats = (train_ds.amp_mean, train_ds.amp_std)
        val_ds = SpectrumSOCDataset("data/_split_val.csv", amp_stats=amp_stats)
        n_freq = train_ds.n_freq

    # Derive frequency labels (from amp_* columns) for interpretability export
    try:
        header_cols = list(pd.read_csv(os.path.join(comb_dir, "train.csv"), nrows=0).columns)
        freq_labels = [c.replace("amp_", "") for c in header_cols[:-1][0::3]]
    except Exception:
        freq_labels = [str(i) for i in range(n_freq)]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ITransformerEncoder(
        d_in=3,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        use_layernorm=True,
        n_freq=n_freq
    ).to(device)

    # Regression: L1 + L2 hybrid
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_rmse = 1e9
    epochs_no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        def loss_fn(pred, target):
            return 0.5 * loss_l1(pred, target) + 0.5 * loss_l2(pred, target)

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_out = evaluate(
            model,
            val_loader,
            device,
            collect_attn=args.save_attn,
            collect_preds=args.save_preds,
        )
        if args.save_attn and args.save_preds:
            mae, rmse, attn_avg, y_val_pred_true, y_val_pred = eval_out
        elif args.save_attn:
            mae, rmse, attn_avg = eval_out
        elif args.save_preds:
            mae, rmse, y_val_pred_true, y_val_pred = eval_out
        else:
            mae, rmse = eval_out
        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_MAE={mae:.4f} | val_RMSE={rmse:.4f}")

        # Append metrics log
        os.makedirs("vis_outputs", exist_ok=True)
        log_path = Path("vis_outputs/training_log.csv")
        row = pd.DataFrame([{ "epoch": epoch, "train_loss": train_loss, "val_mae": mae, "val_rmse": rmse }])
        if log_path.exists():
            row.to_csv(log_path, mode="a", header=False, index=False)
        else:
            row.to_csv(log_path, index=False)

        # Optional: save averaged frequency attention over validation set
        if args.save_attn:
            os.makedirs("vis_outputs", exist_ok=True)
            df_attn = pd.DataFrame({"freq": freq_labels, "weight": attn_avg})
            df_attn.to_csv("vis_outputs/freq_attention_avg.csv", index=False)
            df_attn.to_csv(f"vis_outputs/freq_attention_avg_epoch_{epoch:03d}.csv", index=False)

        # Optional: save validation predictions each epoch
        if args.save_preds:
            # restore to original scale 0-100
            df_pred = pd.DataFrame({
                "y_true": (y_val_pred_true * 100.0),
                "y_pred": (y_val_pred * 100.0),
            })
            df_pred.to_csv(f"vis_outputs/val_preds_epoch_{epoch:03d}.csv", index=False)

        if rmse < best_rmse:
            best_rmse = rmse
            epochs_no_improve = 0
            payload = {
                "model": model.state_dict(),
                "config": vars(args),
                "n_freq": n_freq,
                "epoch": int(epoch),
                "val_mae": float(mae),
                "val_rmse": float(rmse),
                "train_loss": float(train_loss),
            }
            # Save/overwrite best.pt for convenience
            torch.save(payload, "checkpoints/best.pt")
            # Also save an individual, timestamped checkpoint
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            uniq = f"checkpoints/ckpt_epoch{epoch:03d}_rmse{rmse:.4f}_{stamp}.pt"
            torch.save(payload, uniq)
            print(f"  ✓ saved best to checkpoints/best.pt and {uniq}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.33)
    p.add_argument("--save_attn", action="store_true")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()
    main(args)

