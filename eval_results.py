import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def latest_run_dir(root: Path = Path("runs"), prefix: str = "eval-") -> Path:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def load_preds(run_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    preds = []
    for f in sorted(run_dir.glob("*_preds.csv")):
        df = pd.read_csv(f)
        if not set(["y_true", "y_pred"]).issubset(df.columns):
            continue
        preds.append((f.stem.replace("_preds", ""), df["y_true"].values, df["y_pred"].values))
    return preds


def load_attentions(run_dir: Path) -> List[pd.DataFrame]:
    attns = []
    for f in sorted(run_dir.glob("*_attention.csv")):
        df = pd.read_csv(f)
        # Expect columns: freq, weight
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["freq", "weight"]
            attns.append(df)
    return attns


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str, metrics: Optional[dict] = None):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('True SOC (0-1)')
    plt.ylabel('Predicted SOC (0-1)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if metrics:
        text = "\n".join([
            f"MAE:  {metrics['mae']:.4f}",
            f"RMSE: {metrics['rmse']:.4f}",
            f"MSE:  {metrics['mse']:.6f}",
            f"R2:   {metrics['r2']:.4f}",
        ])
        plt.gca().text(0.02, 0.98, text, transform=plt.gca().transAxes,
                       va='top', ha='left', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_attention_heatmap(freqs: List[str], weights: np.ndarray, out_path: Path, title: str):
    arr = weights.reshape(1, -1)
    plt.figure(figsize=(10, 2))
    plt.imshow(arr, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    n = len(freqs)
    step = max(1, n // 20)
    xticks = np.arange(0, n, step)
    plt.xticks(xticks, np.array(freqs)[::step], rotation=45, ha='right')
    plt.yticks([])
    plt.xlabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def average_attentions(attn_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not attn_dfs:
        return None
    # Use first file's ordering; outer-join remaining by freq
    base = attn_dfs[0].copy()
    base.columns = ["freq", "w0"]
    merged = base
    for i, df in enumerate(attn_dfs[1:], start=1):
        tmp = df.copy()
        tmp.columns = ["freq", f"w{i}"]
        merged = pd.merge(merged, tmp, on="freq", how="outer")
    # Fill missing with 0 and compute mean across weight columns
    w_cols = [c for c in merged.columns if c.startswith("w")]
    merged[w_cols] = merged[w_cols].fillna(0.0)
    merged["weight"] = merged[w_cols].mean(axis=1)
    # Restore frequency order: keep the order from the first df; append any new freqs at the end
    order = list(attn_dfs[0]["freq"].astype(str))
    merged["freq"] = merged["freq"].astype(str)
    merged["_order"] = merged["freq"].apply(lambda x: order.index(x) if x in order else len(order))
    merged = merged.sort_values(["_order", "freq"]).drop(columns=["_order"])
    return merged[["freq", "weight"]]


def main():
    ap = argparse.ArgumentParser(description="Visualize eval_infer results: SOC scatter and attention heatmap")
    ap.add_argument('--run-dir', type=str, default=None, help='Path to a runs/eval-* directory')
    ap.add_argument('--per-file', action='store_true', help='Also save per-file scatter plots')
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    assert run_dir is not None and run_dir.exists(), "No evaluation run directory found."

    # Load predictions
    preds = load_preds(run_dir)
    if preds:
        # Per-file scatter
        if args.per_file:
            for name, yt, yp in preds:
                plot_scatter(yt/100.0, yp/100.0, run_dir / f"{name}_scatter.png", f"True vs Predicted SOC ({name})")
        # Aggregated scatter
        all_y = np.concatenate([yt for _, yt, _ in preds]) / 100.0
        all_p = np.concatenate([yp for _, _, yp in preds]) / 100.0
        # Metrics in decimal (0-1)
        mae = mean_absolute_error(all_y, all_p)
        mse = mean_squared_error(all_y, all_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_y, all_p)
        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
        # Save metrics (decimals)
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics_overall.csv", index=False)
        print(f"Overall metrics -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, MSE: {mse:.6f}, R2: {r2:.4f}")
        plot_scatter(all_y, all_p, run_dir / "all_scatter.png", "True vs Predicted SOC (All)", metrics)
    else:
        print("No *_preds.csv found in run directory.")

    # Load attentions and average
    attn_dfs = load_attentions(run_dir)
    if attn_dfs:
        avg = average_attentions(attn_dfs)
        if avg is not None:
            avg.to_csv(run_dir / "avg_attention.csv", index=False)
            plot_attention_heatmap(avg["freq"].astype(str).tolist(), avg["weight"].values,
                                   run_dir / "avg_attention_heatmap.png",
                                   "Average Frequency Attention (All)")
    else:
        print("No *_attention.csv found in run directory.")


if __name__ == '__main__':
    main()
