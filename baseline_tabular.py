import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
try:
    import xgboost as xgb
    _has_xgb = True
except Exception:
    _has_xgb = False

def set_seed(seed: int):
    import os, random, numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_combined_frames(path: Path):
    # Accept a single file or a folder with combined_*.csv (or val.csv fallback)
    if path.is_file():
        return [path]
    files = sorted(path.glob('combined_*.csv'))
    if not files and (path / 'val.csv').exists():
        files = [path / 'val.csv']
    return files


def amp_only_scaling(X: np.ndarray, fit: bool, stats=None):
    """Scale only amplitude channels in triplets [amp, sin, cos].
    X: [N, 3*L]
    Returns X_scaled, (mean,std) if fit else X_scaled, stats
    """
    assert X.shape[1] % 3 == 0, "特征维应为3的倍数(amp,sin,cos)"
    L = X.shape[1] // 3
    amp = X[:, 0::3]
    if fit:
        mean = amp.mean(axis=0, keepdims=True)
        std = amp.std(axis=0, keepdims=True) + 1e-6
    else:
        assert stats is not None
        mean, std = stats
    X_scaled = X.copy()
    X_scaled[:, 0::3] = (amp - mean) / std
    return X_scaled, (mean, std)


def build_dataset(files):
    frames = []
    groups = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        groups.extend([f.stem] * len(df))
    all_df = pd.concat(frames, ignore_index=True)
    cols = list(all_df.columns)
    assert cols[-1].lower() == 'soc'
    X = all_df[cols[:-1]].astype(np.float32).values
    y = all_df[cols[-1]].astype(np.float32).values / 100.0  # 0-1
    groups = np.array(groups)
    return X, y, groups, cols


def evaluate_and_save(y_true, y_pred, out_dir: Path, stem: str):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Save metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{ 'mae': mae, 'rmse': rmse, 'mse': mse, 'r2': r2 }]).to_csv(out_dir / f'{stem}_metrics.csv', index=False)
    # Save preds (0-100 scale for readability)
    pd.DataFrame({ 'y_true': y_true * 100.0, 'y_pred': y_pred * 100.0 }).to_csv(out_dir / f'{stem}_preds.csv', index=False)
    return mae, rmse, r2


def main():
    ap = argparse.ArgumentParser(description='Tabular baselines (MLP/XGBoost) on combined features')
    ap.add_argument('--data', type=str, default='data/combined', help='Dir with combined_*.csv or a single CSV file')
    ap.add_argument('--model', type=str, default='xgb', choices=['mlp', 'xgb'], help='Baseline model type')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-ratio', type=float, default=0.33)
    ap.add_argument('--tag', type=str, default=None)
    # MLP params
    ap.add_argument('--mlp-hidden', type=str, default='256,128', help='Comma-separated hidden sizes for MLP')
    ap.add_argument('--mlp-max-iter', type=int, default=500)
    # XGB params
    ap.add_argument('--xgb-rounds', type=int, default=2000)
    ap.add_argument('--xgb-lr', type=float, default=0.05)
    ap.add_argument('--xgb-depth', type=int, default=6)
    ap.add_argument('--xgb-subsample', type=float, default=0.8)
    ap.add_argument('--xgb-colsample', type=float, default=0.8)
    ap.add_argument('--xgb-es', type=int, default=100, help='early_stopping_rounds')
    args = ap.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    files = load_combined_frames(data_path)
    assert files, f'No data found at {data_path}'

    X, y, groups, cols = build_dataset(files)

    # Group-aware split if we have multiple stems
    stems = np.unique(groups)
    if len(stems) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
        train_idx, val_idx = next(gss.split(X, groups=groups))
    else:
        train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=args.test_ratio, random_state=args.seed)

    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale amp only (keep sin/cos untouched)
    X_train, amp_stats = amp_only_scaling(X_train_raw, fit=True)
    X_val, _ = amp_only_scaling(X_val_raw, fit=False, stats=amp_stats)

    stamp = args.tag or datetime.now().strftime('%Y%m%d-%H%M%S')
    # Save into a dedicated folder compared_model/<model>-<timestamp>
    out_dir = Path('compared_model') / f'{args.model}-{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == 'mlp':
        hidden = tuple(int(x) for x in args.mlp_hidden.split(',') if x.strip())
        model = MLPRegressor(hidden_layer_sizes=hidden,
                             activation='relu', solver='adam',
                             max_iter=args.mlp_max_iter,
                             random_state=args.seed,
                             early_stopping=True,
                             n_iter_no_change=20,
                             validation_fraction=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae, rmse, r2 = evaluate_and_save(y_val, y_pred, out_dir, 'mlp')
        print(f"MLP -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    else:
        # try:
        #     from xgboost import XGBRegressor
        # except Exception as e:
        #     raise ImportError("需要安装 xgboost 才能使用 --model xgb (pip install xgboost)") from e

        model = xgb.XGBRegressor(
            n_estimators=args.xgb_rounds,
            learning_rate=args.xgb_lr,
            max_depth=args.xgb_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=args.seed,
            tree_method='hist',
            eval_metric='rmse',
        )
        try:
            cb = [xgb.callback.EarlyStopping(rounds=args.xgb_es, save_best=True, maximize=False)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=cb,
            )
        except TypeError:
            # Older sklearn wrapper: try early_stopping_rounds argument
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=args.xgb_es,
                )
            except TypeError:
                # Very old version: no early stopping available in fit -> fallback to plain fit
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
        y_pred = model.predict(X_val)
        mae, rmse, r2 = evaluate_and_save(y_val, y_pred, out_dir, 'xgb')
        print(f"XGB -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")


if __name__ == '__main__':
    main()
