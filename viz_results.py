import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def latest_file(pattern: str):
    files = sorted(Path("vis_outputs").glob(pattern))
    return files[-1] if files else None


def plot_scatter(preds_csv: Path, out_path: Path):
    df = pd.read_csv(preds_csv)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('True SOC')
    plt.ylabel('Predicted SOC')
    plt.title('True vs Predicted SOC (Validation)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved scatter: {out_path}")


def plot_attention_heatmap(attn_csv: Path, out_path: Path):
    df = pd.read_csv(attn_csv)
    # Expect columns: freq, weight
    freqs = df[df.columns[0]].astype(str).values
    weights = df[df.columns[1]].values.reshape(1, -1)

    plt.figure(figsize=(10, 2))
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    # Tick sparsity for readability
    n = len(freqs)
    step = max(1, n // 20)
    xticks = np.arange(0, n, step)
    plt.xticks(xticks, freqs[::step], rotation=45, ha='right')
    plt.yticks([])
    plt.xlabel('Frequency')
    plt.title('Average Frequency Attention (Validation)')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', type=str, default=None, help='Path to validation preds CSV (y_true,y_pred)')
    ap.add_argument('--attn', type=str, default=None, help='Path to attention CSV (freq,weight)')
    args = ap.parse_args()

    preds_csv = Path(args.preds) if args.preds else latest_file('val_preds_epoch_*.csv')
    attn_csv = Path(args.attn) if args.attn else latest_file('freq_attention_avg_epoch_*.csv')
    # fallback to the latest single attention file
    if attn_csv is None:
        attn_csv = Path('vis_outputs/freq_attention_avg.csv') if Path('vis_outputs/freq_attention_avg.csv').exists() else None

    if preds_csv and preds_csv.exists():
        plot_scatter(preds_csv, Path('vis_outputs/soc_scatter.png'))
    else:
        print('No prediction CSV found. Enable --save_preds in run.py to generate them.')

    if attn_csv and attn_csv.exists():
        plot_attention_heatmap(attn_csv, Path('vis_outputs/freq_attention_heatmap.png'))
    else:
        print('No attention CSV found. Enable --save_attn in run.py to generate them.')


if __name__ == '__main__':
    main()

