import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run import ITransformerEncoder, SpectrumSOCDataset  # reuse classes


@torch.no_grad()
def evaluate(model, loader, device, collect_attn: bool = True):
    model.eval()
    ys, ps = [], []
    attn_list = [] if collect_attn else None
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
        if collect_attn and getattr(model, 'last_freq_attn', None) is not None:
            attn_list.append(model.last_freq_attn.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    attn_avg = None
    if collect_attn and attn_list:
        attn_avg = np.concatenate(attn_list, axis=0).mean(axis=0)
    return y, p, attn_avg


def derive_freq_labels(csv_path: Path, n_freq: int):
    try:
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        # Expect amp_* first of each triplet
        labels = [c.replace("amp_", "") for c in cols[:-1][0::3]]
        if len(labels) == n_freq:
            return labels
    except Exception:
        pass
    return [str(i) for i in range(n_freq)]


def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoint on combined CSVs and save results per run")
    ap.add_argument('--ckpt', type=str, default='checkpoints1/best_original.pt', help='Path to checkpoint')
    ap.add_argument('--data-dir', type=str, default='data/combined/', help='Directory with combined_*.csv or a single CSV file path')
    ap.add_argument('--train-csv', type=str, default=None, help='Optional train.csv to compute amp normalization stats')
    ap.add_argument('--out-root', type=str, default='runs', help='Root folder for outputs')
    ap.add_argument('--tag', type=str, default=None, help='Optional tag for output folder name')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--no-attn', action='store_true', help='Disable attention collection')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt.get('config', {})
    n_freq = ckpt.get('n_freq')
    assert n_freq is not None, 'Checkpoint missing n_freq'

    attn_type = cfg.get('attn_type', 'vanilla')
    model = ITransformerEncoder(
        d_in=3,
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('nhead', 8),
        num_layers=cfg.get('layers', 4),
        dim_feedforward=cfg.get('ffn', 256),
        dropout=cfg.get('dropout', 0.1),
        use_layernorm=True,
        n_freq=n_freq,
        attn_type=attn_type,
    ).to(device)
    model.load_state_dict(ckpt['model'])

    data_path = Path(args.data_dir)
    # Accept a single file path or a directory
    if data_path.is_file():
        files = [data_path]
    else:
        files = sorted(data_path.glob('combined_*.csv'))
        if not files and (data_path / 'val.csv').exists():
            files = [data_path / 'val.csv']
    assert files, f'No evaluation files found in {data_path}'

    # Output directory
    stamp = args.tag or datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out_root) / f'eval-{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get amp normalization stats
    amp_stats = None
    if args.train_csv and Path(args.train_csv).exists():
        tmp_train = SpectrumSOCDataset(args.train_csv)
        amp_stats = (tmp_train.amp_mean, tmp_train.amp_std)

    all_preds = []
    # Evaluate each file separately and save
    for f in files:
        ds = SpectrumSOCDataset(str(f), amp_stats=amp_stats)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        y_true, y_pred, attn_avg = evaluate(model, loader, device, collect_attn=not args.no_attn)

        # Save predictions (restore 0-100 scale)
        pred_path = out_dir / f'{f.stem}_preds.csv'
        pd.DataFrame({
            'y_true': y_true * 100.0,
            'y_pred': y_pred * 100.0,
        }).to_csv(pred_path, index=False)

        # Save attention (with frequency labels)
        if attn_avg is not None:
            freqs = derive_freq_labels(f, ds.n_freq)
            pd.DataFrame({'freq': freqs, 'weight': attn_avg}).to_csv(out_dir / f'{f.stem}_attention.csv', index=False)

        all_preds.append((f.name, y_true, y_pred))

    # Summarize metrics
    rows = []
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    for name, y, p in all_preds:
        mae = mean_absolute_error(y, p)
        rmse = mean_squared_error(y, p, squared=False)
        rows.append({'file': name, 'mae': mae, 'rmse': rmse})
    pd.DataFrame(rows).to_csv(out_dir / 'summary.csv', index=False)
    print(f'Saved results to {out_dir}')


if __name__ == '__main__':
    main()
