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
    soc = df.iloc[2, 1:].to_numpy()
    freqs = df.iloc[3:, 0].to_numpy()
    vals = df.iloc[3:, 1:].to_numpy()
    return soc, freqs, vals


def _choose_ticks(n: int, max_ticks: int = 10, step: int = 0):
    """Choose sparse tick indices for an axis of length n.

    Priority: use max_ticks if >0, else use step if >0, else auto (≈10 ticks).
    Returns a 1D array of indices (int) within [0, n).
    """
    if n <= 0:
        return np.array([], dtype=int)
    if max_ticks and max_ticks > 0:
        k = max(2, min(max_ticks, n))
        return np.unique(np.linspace(0, n - 1, k).astype(int))
    if step and step > 0:
        return np.arange(0, n, step, dtype=int)
    # default ~10 ticks
    k = max(2, min(10, n))
    return np.unique(np.linspace(0, n - 1, k).astype(int))


def plot_heatmap(soc, freqs, vals, out_path: Path, title: str,
                 soc_tick_step: int = 0, freq_tick_step: int = 0,
                 soc_max_ticks: int = 10, freq_max_ticks: int = 12,
                 cmap: str = 'viridis', value_label: str = 'Value', invert_x: bool = False):
    """Plot a SOC (x) vs Frequency (y) heatmap of vals[freq, soc] with sparse ticks."""
    # Ensure numeric
    soc = np.array(soc)
    freqs = np.array(freqs)
    vals = np.array(vals, dtype=float)

    plt.figure(figsize=(10, 5))
    # origin='lower' so lower frequencies at bottom
    im = plt.imshow(vals, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(im, label=value_label)

    # X ticks for SOC
    n_soc = soc.shape[0]
    x_idx = _choose_ticks(n_soc, max_ticks=soc_max_ticks, step=soc_tick_step)
    plt.xticks(x_idx, soc[x_idx])
    plt.xlabel('SOC')

    # Y ticks for freq
    n_freq = freqs.shape[0]
    y_idx = _choose_ticks(n_freq, max_ticks=freq_max_ticks, step=freq_tick_step)
    plt.yticks(y_idx, freqs[y_idx])
    plt.ylabel('Frequency')

    if invert_x:
        plt.gca().invert_xaxis()

    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def infer_kind_from_path(p: Path) -> str:
    stem = str(p).lower()
    if 'phase' in stem:
        return 'phase'
    if 'magnitude' in stem or 'amplitude' in stem:
        return 'magnitude'
    return 'values'


def process_path(input_path: Path, out_dir: Path,
                 soc_tick: int, freq_tick: int,
                 soc_max: int, freq_max: int,
                 invert_x: bool):
    if input_path.is_file():
        files = [input_path]
    else:
        # look for both magnitude and phase Excel files
        files = sorted(list(input_path.glob('**/*.xlsx')))
    if not files:
        print(f"No .xlsx files found under {input_path}")
        return

    for f in files:
        try:
            soc, freqs, vals = load_excel_matrix(f)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue
        kind = infer_kind_from_path(f)
        out_path = out_dir / f"heatmap_{kind}_{f.stem}.png"
        title = f"{kind.capitalize()} SOC-Frequency Heatmap: {f.stem}"
        value_label = 'Phase' if kind == 'phase' else ('Amplitude' if kind == 'magnitude' else 'Value')
        plot_heatmap(
            soc, freqs, vals,
            out_path=out_path,
            title=title,
            soc_tick_step=soc_tick,
            freq_tick_step=freq_tick,
            soc_max_ticks=soc_max,
            freq_max_ticks=freq_max,
            value_label=value_label,
            invert_x=invert_x,
        )


def main():
    ap = argparse.ArgumentParser(description='SOC (x) vs Frequency (y) heatmap for magnitude/phase Excel files')
    ap.add_argument('--input', type=str, default=r"C:\Users\86182\Desktop\SOC_DATA\0.3c_Cycle1-Cycle6",
                    help='Path to a folder or a single Excel file')
    ap.add_argument('--out-dir', type=str, default='vis_outputs', help='Output directory for images')
    ap.add_argument('--soc-tick', type=int, default=0, help='Tick step along SOC axis (0=auto)')
    ap.add_argument('--freq-tick', type=int, default=0, help='Tick step along frequency axis (0=auto)')
    ap.add_argument('--soc-max', type=int, default=10, help='Max number of SOC ticks (evenly spaced)')
    ap.add_argument('--freq-max', type=int, default=12, help='Max number of frequency ticks (evenly spaced)')
    ap.add_argument('--invert-x', action='store_true', help='Invert x-axis (SOC)')
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    process_path(
        input_path,
        out_dir,
        soc_tick=args.soc_tick,
        freq_tick=args.freq_tick,
        soc_max=args.soc_max,
        freq_max=args.freq_max,
        invert_x=args.invert_x,
    )


if __name__ == '__main__':
    main()
