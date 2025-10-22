import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_excel_matrix(xlsx: Path):
    """Load one Excel file in the known layout and return (soc, freqs, values).

    Layout:
      - Row 0: headers (Time, SOC columns labels)
      - Row 1: Ah (unused)
      - Row 2: SOC labels (targets)
      - Rows 3+: column 0 is frequency, remaining columns are values
    Returns:
      soc: np.ndarray [n_samples]
      freqs: np.ndarray [n_freq]
      vals: np.ndarray [n_freq, n_samples]
    """
    df = pd.read_excel(xlsx, header=None)
    soc = pd.to_numeric(df.iloc[2, 1:], errors='coerce').to_numpy()
    freqs = pd.to_numeric(df.iloc[3:, 0], errors='coerce').to_numpy()
    # Convert DataFrame to numeric per-column to avoid pandas error
    vals = df.iloc[3:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()
    return soc, freqs, vals


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = min(w, len(x))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode='same')


def pick_frequencies(soc: np.ndarray, freqs: np.ndarray, vals: np.ndarray,
                     topk: int = 0, uniform_n: int = 0) -> np.ndarray:
    """Return indices of selected frequencies to plot.

    - If topk > 0: pick top-k by absolute Pearson correlation with SOC.
    - Else if uniform_n > 0: pick evenly spaced frequencies.
    - Else: default top-9.
    """
    n_freq = vals.shape[0]
    if n_freq == 0:
        return np.array([], dtype=int)
    # Default: plot ALL frequencies when no selection hint provided
    if topk <= 0 and uniform_n <= 0:
        return np.arange(n_freq, dtype=int)
    if uniform_n and uniform_n > 0:
        k = min(uniform_n, n_freq)
        return np.unique(np.linspace(0, n_freq - 1, k).astype(int))
    # compute abs correlation per frequency
    y = (soc - np.nanmean(soc))
    y_std = np.nanstd(soc) + 1e-9
    corrs = []
    for i in range(n_freq):
        xi = vals[i, :]
        xi = xi - np.nanmean(xi)
        denom = (np.nanstd(xi) + 1e-9) * y_std
        corr = float(np.nansum(xi * y) / (len(y) * denom))
        corrs.append(abs(corr))
    order = np.argsort(-np.array(corrs))
    k = min(topk, n_freq)
    return order[:k]


def plot_overlay(soc, freqs, vals, idx_sel: np.ndarray, out_path: Path,
                 title: str, smooth: int = 1, invert_x: bool = False,
                 soc_max_ticks: int = 10):
    plt.figure(figsize=(10, 5))
    # choose sparse x ticks
    n_soc = len(soc)
    k = max(2, min(soc_max_ticks, n_soc))
    x_idx = np.unique(np.linspace(0, n_soc - 1, k).astype(int))
    for i in idx_sel:
        y = vals[i, :]
        if smooth and smooth > 1:
            y = moving_average(y, smooth)
        plt.plot(np.arange(n_soc), y, linewidth=1.2, alpha=0.9, label=f"{freqs[i]}")
    plt.xticks(x_idx, soc[x_idx])
    plt.xlabel('SOC')
    plt.ylabel('Value')
    plt.title(title)
    if invert_x:
        plt.gca().invert_xaxis()
    # keep legend compact (hide if too many lines)
    if len(idx_sel) <= 20:
        plt.legend(title='Freq', bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_grid(soc, freqs, vals, idx_sel: np.ndarray, out_path: Path,
              title: str, rows: int = 3, cols: int = 3,
              smooth: int = 1, invert_x: bool = False,
              soc_max_ticks: int = 5):
    n = len(idx_sel)
    rows = max(1, rows)
    cols = max(1, cols)
    plt.figure(figsize=(cols * 3.2, rows * 2.4))
    n_soc = len(soc)
    k = max(2, min(soc_max_ticks, n_soc))
    x_idx = np.unique(np.linspace(0, n_soc - 1, k).astype(int))
    for j in range(min(n, rows * cols)):
        i = idx_sel[j]
        ax = plt.subplot(rows, cols, j + 1)
        y = vals[i, :]
        if smooth and smooth > 1:
            y = moving_average(y, smooth)
        ax.plot(np.arange(n_soc), y, linewidth=1.2)
        ax.set_title(f"f={freqs[i]}", fontsize=9)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(soc[x_idx], rotation=30, fontsize=7)
        if invert_x:
            ax.invert_xaxis()
        if j % cols == 0:
            ax.set_ylabel('Value')
        else:
            ax.set_yticklabels([])
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description='Plot per-frequency value vs SOC trends')
    ap.add_argument('--input', type=str, default=r"C:\Users\86182\Desktop\SOC_DATA\0.3c_Cycle1-Cycle6",
                    help='Path to a folder or a single Excel file')
    ap.add_argument('--out-dir', type=str, default='vis_outputs', help='Output directory for images')
    ap.add_argument('--topk', type=int, default=9, help='Select top-k frequencies by |corr(value,SOC)|')
    ap.add_argument('--uniform', type=int, default=0, help='If >0, ignore topk and plot evenly sampled N frequencies')
    ap.add_argument('--smooth', type=int, default=1, help='Moving-average window size (1=no smoothing)')
    ap.add_argument('--grid', action='store_true', help='Plot as small-multiple grid instead of single overlay')
    ap.add_argument('--rows', type=int, default=3)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--soc-max', type=int, default=10, help='Max number of SOC ticks')
    ap.add_argument('--invert-x', action='store_true')
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    files = [input_path] if input_path.is_file() else sorted(list(input_path.glob('**/*.xlsx')))
    if not files:
        print(f"No .xlsx files found under {input_path}")
        return

    for f in files:
        try:
            soc, freqs, vals = load_excel_matrix(f)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue
        # select frequencies
        idx_sel = pick_frequencies(soc, freqs, vals, topk=args.topk, uniform_n=args.uniform)
        kind = 'phase' if 'phase' in str(f).lower() else ('magnitude' if 'magnitude' in str(f).lower() or 'amplitude' in str(f).lower() else 'values')
        title = f"{kind.capitalize()} Trends vs SOC: {f.stem}"
        if args.grid:
            out_path = out_dir / f"trends_grid_{kind}_{f.stem}.png"
            plot_grid(soc, freqs, vals, idx_sel, out_path, title,
                      rows=args.rows, cols=args.cols,
                      smooth=args.smooth, invert_x=args.invert_x,
                      soc_max_ticks=args.soc_max)
        else:
            out_path = out_dir / f"trends_overlay_{kind}_{f.stem}.png"
            plot_overlay(soc, freqs, vals, idx_sel, out_path, title,
                         smooth=args.smooth, invert_x=args.invert_x,
                         soc_max_ticks=args.soc_max)


if __name__ == '__main__':
    main()
