import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_phase_excel(infile: Path):
    """Parse one phase Excel file with the expected layout.

    Layout:
      - Row 0: headers (Time, 0, 43, ...)
      - Row 1: Ah (unused)
      - Row 2: SOC labels (targets)
      - Row 3+: column 0 is frequency, remaining columns are phase values
    Returns freqs (1D), phase_matrix (2D [freqs, samples]), soc (1D [samples])
    """
    df_raw = pd.read_excel(infile, header=None)

    soc_labels = df_raw.iloc[2, 1:].to_numpy()
    freqs = df_raw.iloc[3:, 0].to_numpy()
    phase_matrix = df_raw.iloc[3:, 1:].to_numpy()
    return freqs, phase_matrix, soc_labels


def build_raw_df(freqs: np.ndarray, phase_matrix: np.ndarray, soc_labels: np.ndarray) -> pd.DataFrame:
    """Return DataFrame with columns=freqs and final column 'soc'."""
    X = pd.DataFrame(phase_matrix.T, columns=freqs)
    X["soc"] = pd.to_numeric(soc_labels, errors="coerce")
    return X


def build_trig_df(
    freqs: np.ndarray,
    phase_matrix: np.ndarray,
    soc_labels: np.ndarray,
    phase_unit: str = "deg",
) -> pd.DataFrame:
    """Return DataFrame with sin/cos transformed phase columns and final 'soc'.

    - If phase_unit='deg', convert to radians first; if 'rad', use as-is.
    Columns are named 'sin_<freq>' and 'cos_<freq>'.
    """
    if phase_unit not in {"deg", "rad"}:
        raise ValueError("phase_unit must be 'deg' or 'rad'")

    radians = np.deg2rad(phase_matrix) if phase_unit == "deg" else phase_matrix
    sin_mat = np.sin(radians)  # shape [freqs, samples]
    cos_mat = np.cos(radians)

    # Build DataFrame with columns ordered: all sin_*, then all cos_*
    sin_cols = [f"sin_{f}" for f in freqs]
    cos_cols = [f"cos_{f}" for f in freqs]
    data = np.hstack((sin_mat.T, cos_mat.T))  # [samples, 2*freqs]
    X = pd.DataFrame(data, columns=sin_cols + cos_cols)
    X["soc"] = pd.to_numeric(soc_labels, errors="coerce")
    return X


def process_file(infile: Path, out_dir: Path, prefix: str, phase_unit: str) -> None:
    freqs, phase_matrix, soc_labels = parse_phase_excel(infile)

    # Raw phase
    df_raw = build_raw_df(freqs, phase_matrix, soc_labels)
    out_raw = out_dir / f"{prefix}raw_{infile.stem}.csv"
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(out_raw, index=False)
    print(f"已生成: {out_raw}")

    # Trig transformed
    df_trig = build_trig_df(freqs, phase_matrix, soc_labels, phase_unit=phase_unit)
    out_trig = out_dir / f"{prefix}trig_{infile.stem}.csv"
    df_trig.to_csv(out_trig, index=False)
    print(f"已生成: {out_trig}")


def process_path(input_path: Path, out_dir: Path, pattern: str, prefix: str, phase_unit: str) -> None:
    if input_path.is_file():
        process_file(input_path, out_dir, prefix, phase_unit)
        return
    if input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        if not files:
            print(f"目录内未找到匹配文件: {input_path} ({pattern})")
            return
        for f in files:
            process_file(f, out_dir, prefix, phase_unit)
        print(f"完成，共处理 {len(files)} 个文件 -> {out_dir}")
        return
    raise FileNotFoundError(f"路径不存在: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Process phase Excel files to CSV (raw and trig encoded)")
    parser.add_argument(
        "input",
        nargs="?",
        default=r"C:\\Users\\86182\\Desktop\\SOC_DATA\\0.3c_Cycle1-Cycle6\\phase",
        help="输入路径:单个xlsx文件或包含xlsx的目录",
    )
    parser.add_argument("--out-dir", default="data/phase", help="输出目录，默认 data/phase")
    parser.add_argument("--pattern", default="*.xlsx", help="目录模式匹配，默认*.xlsx")
    parser.add_argument("--prefix", default="phase_", help="输出文件名前缀，例如 phase_")
    parser.add_argument(
        "--phase-unit",
        choices=["deg", "rad"],
        default="deg",
        help="相位单位:deg(角度) 或 rad(弧度)，默认 deg",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    process_path(input_path, out_dir, pattern=args.pattern, prefix=args.prefix, phase_unit=args.phase_unit)


if __name__ == "__main__":
    main()

