import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def excel_to_csv(infile: Path, out_csv: Path) -> None:
    """Convert one Excel file (specific row layout) to CSV with soc column.

    Layout expectation:
      - Row 0: headers (Time, 0, 43, ...)
      - Row 1: Ah info (unused)
      - Row 2: SOC labels (used)
      - Row 3+: frequency values in col 0, amplitudes in remaining columns
    """
    df_raw = pd.read_excel(infile, header=None)

    soc_labels = df_raw.iloc[2, 1:].to_numpy()
    freqs = df_raw.iloc[3:, 0].to_numpy()
    amp_matrix = df_raw.iloc[3:, 1:].to_numpy()

    X = pd.DataFrame(amp_matrix.T, columns=freqs)
    X["soc"] = pd.to_numeric(soc_labels, errors="coerce")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(out_csv, index=False)


def process_path(input_path: Path, out_dir: Path, pattern: str = "*.xlsx", prefix: str = "train_") -> None:
    """Process either a single file or all Excel files in a directory.

    - If input_path is a file: write to out_dir/prefix<stem>.csv
    - If input_path is a dir:  find all files matching pattern and write each
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        out_csv = out_dir / f"{prefix}{input_path.stem}.csv"
        excel_to_csv(input_path, out_csv)
        print(f"已生成: {out_csv}")
        return

    if input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        if not files:
            print(f"目录内未找到匹配文件: {input_path} ({pattern})")
            return
        for f in files:
            out_csv = out_dir / f"{prefix}{f.stem}.csv"
            excel_to_csv(f, out_csv)
            print(f"已生成: {out_csv}")
        print(f"完成，共处理 {len(files)} 个文件 -> {out_dir}")
        return

    raise FileNotFoundError(f"路径不存在: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Excel files to CSV with soc column.")
    parser.add_argument(
        "input",
        nargs="?",
        default=r"C:\\Users\\86182\\Desktop\\SOC_DATA\\0.3c_Cycle1-Cycle6\\magnitude",
        help="输入路径: 单个xlsx文件或包含xlsx的目录",
    )
    parser.add_argument("--out-dir", default="data", help="输出目录")
    parser.add_argument("--pattern", default="*.xlsx", help="目录模式匹配，默认*.xlsx")
    parser.add_argument("--prefix", default="train_", help="输出文件名前缀，例如 train_")

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    process_path(input_path, out_dir, pattern=args.pattern, prefix=args.prefix)


if __name__ == "__main__":
    main()
