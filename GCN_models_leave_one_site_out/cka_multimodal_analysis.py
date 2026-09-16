#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linear CKA analysis for BrainANet-SwinViT and RSFC representations."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t = None


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_columns(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_feature_columns(
        df: pd.DataFrame,
        id_col: str,
        group_col: Optional[str],
        label_col: Optional[str],
        feature_prefix: Optional[str],
        exclude_cols: Sequence[str],
) -> list[str]:
    excluded = {id_col, *exclude_cols}
    if group_col:
        excluded.add(group_col)
    if label_col:
        excluded.add(label_col)

    if feature_prefix:
        columns = [
            col for col in df.columns
            if str(col).startswith(feature_prefix) and col not in excluded
        ]
    else:
        columns = [
            col for col in df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]
    if not columns:
        raise ValueError("No numeric feature columns were found.")
    return columns


def validate_keys(df: pd.DataFrame, id_col: str, group_col: Optional[str], name: str) -> None:
    keys = [id_col] + ([group_col] if group_col else [])
    missing = [col for col in keys if col not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")
    if df.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicated subject/run keys.")


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    mean = np.mean(matrix, axis=0, keepdims=True)
    std = np.std(matrix, axis=0, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (matrix - mean) / std


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA. X and Y may have different feature dimensions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and Y must contain the same subjects.")
    if x.shape[0] < 3:
        return np.nan
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Representations contain NaN or infinite values.")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = np.sum(xy * xy)
    denominator = math.sqrt(np.sum(xx * xx) * np.sum(yy * yy))
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def bootstrap_cka(
        x: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int,
        rng: np.random.Generator,
) -> np.ndarray:
    n = x.shape[0]
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        value = linear_cka(x[idx], y[idx])
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def permutation_cka(
        x: np.ndarray,
        y: np.ndarray,
        n_permutations: int,
        rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        values[i] = linear_cka(x, y[rng.permutation(y.shape[0])])
    return values


def percentile_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = 1.0 - level
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def align_group(
        brain_df: pd.DataFrame,
        rsfc_df: pd.DataFrame,
        brain_features: Sequence[str],
        rsfc_features: Sequence[str],
        id_col: str,
        group_col: Optional[str],
        group_value: object,
        zscore: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if group_col:
        brain_part = brain_df.loc[brain_df[group_col] == group_value].copy()
        rsfc_part = rsfc_df.loc[rsfc_df[group_col] == group_value].copy()
    else:
        brain_part = brain_df.copy()
        rsfc_part = rsfc_df.copy()

    brain_part[id_col] = brain_part[id_col].astype(str)
    rsfc_part[id_col] = rsfc_part[id_col].astype(str)
    ids = sorted(set(brain_part[id_col]) & set(rsfc_part[id_col]))
    if len(ids) < 3:
        raise ValueError(f"Group {group_value!r} has fewer than three matched subjects.")

    brain_part = brain_part.set_index(id_col).loc[ids]
    rsfc_part = rsfc_part.set_index(id_col).loc[ids]
    x = brain_part.loc[:, brain_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = rsfc_part.loc[:, rsfc_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if not valid.all():
        warnings.warn(f"Removed {(~valid).sum()} subjects with invalid values.")
        x, y = x[valid], y[valid]
        ids = [sid for sid, keep in zip(ids, valid) if keep]

    if zscore:
        x, y = zscore_columns(x), zscore_columns(y)
    return x, y, ids


def aggregate_t_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    alpha = 1.0 - level
    if student_t is not None:
        critical = float(student_t.ppf(1.0 - alpha / 2.0, df=values.size - 1))
    else:
        critical = 1.96
    return mean - critical * sem, mean + critical * sem


# def plot_results(per_group: pd.DataFrame, output_dir: Path, confidence_level: float) -> None:
#     fig, ax = plt.subplots(figsize=(7.0 * 0.75, 4.8 * 0.75))
#     x = np.arange(len(per_group))
#     cka = per_group["CKA"].to_numpy(float)
#     low = per_group["Bootstrap_CI_Lower"].to_numpy(float)
#     high = per_group["Bootstrap_CI_Upper"].to_numpy(float)
#     yerr = np.vstack([cka - low, high - cka])
#     ax.errorbar(x, cka, yerr=yerr, fmt="o", capsize=4, linewidth=1.2)
#     ax.set_xticks(x)
#     ax.set_xticklabels(per_group["Group"].astype(str), rotation=30, ha="right")
#     ax.set_ylabel("Linear CKA")
#     ax.set_xlabel("")
#     ax.set_ylim(0.0, 1.02)
#     ax.set_title(f"Cross-modal representation similarity ({int(confidence_level * 100)}% bootstrap CI)")
#     ax.grid(axis="y", linestyle="--", alpha=0.35)
#     fig.tight_layout()
#     fig.savefig(output_dir / "cka_by_group.png", dpi=300, bbox_inches="tight")
#     fig.savefig(output_dir / "cka_by_group.svg", bbox_inches="tight")
#     plt.close(fig)
def save_plot(
        per_group: pd.DataFrame,
        output_dir: Path,
        confidence_level: float,
) -> None:
    """
    绘制跨模态CKA结果。

    主图局部放大低CKA区间，以展示不同重复实验的点估计和95% CI；
    插图保留0--1的完整CKA范围，避免截断纵轴造成视觉误导。
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    data = per_group.copy()

    required_columns = [
        "Group",
        "CKA",
        "Bootstrap_CI_Lower",
        "Bootstrap_CI_Upper",
    ]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"缺少必要列：{missing_columns}")

    # 转换为数值并删除无效记录
    for column in [
        "CKA",
        "Bootstrap_CI_Lower",
        "Bootstrap_CI_Upper",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(subset=[
        "CKA",
        "Bootstrap_CI_Lower",
        "Bootstrap_CI_Upper",
    ]).reset_index(drop=True)

    if data.empty:
        raise ValueError("没有可用于绘图的有效CKA结果。")

    x_positions = np.arange(len(data))
    cka_values = data["CKA"].to_numpy(dtype=float)
    ci_lower = data["Bootstrap_CI_Lower"].to_numpy(dtype=float)
    ci_upper = data["Bootstrap_CI_Upper"].to_numpy(dtype=float)

    lower_error = np.maximum(cka_values - ci_lower, 0)
    upper_error = np.maximum(ci_upper - cka_values, 0)
    error_values = np.vstack([lower_error, upper_error])

    mean_cka = float(np.mean(cka_values))
    std_cka = (
        float(np.std(cka_values, ddof=1))
        if len(cka_values) >= 2
        else 0.0
    )

    # 根据CI上限自动确定主图范围
    maximum_upper_ci = float(np.nanmax(ci_upper))
    zoom_upper = min(
        0.45,
        max(0.10, np.ceil(maximum_upper_ci * 1.20 / 0.05) * 0.05),
    )

    figure, axis = plt.subplots(figsize=(7.5 * 0.75, 5.2 * 0.75))

    # 95% CI
    axis.errorbar(
        x_positions,
        cka_values,
        yerr=error_values,
        fmt="o",
        markersize=7,
        capsize=5,
        capthick=1.3,
        elinewidth=1.3,
        linewidth=1.3,
        label=f"CKA with {int(confidence_level * 100)}% bootstrap CI",
    )

    # 多次重复实验的平均CKA
    axis.axhline(
        mean_cka,
        linestyle="--",
        linewidth=1.3,
        label=f"Mean CKA = {mean_cka:.3f}",
    )

    # 平均值±SD区间
    lower_band = max(0.0, mean_cka - std_cka)
    upper_band = min(1.0, mean_cka + std_cka)

    axis.axhspan(
        lower_band,
        upper_band,
        alpha=0.12,
        label=f"Mean ± SD = {mean_cka:.3f} ± {std_cka:.3f}",
    )

    # 标注每个点的CKA值
    label_offset = zoom_upper * 0.025

    for x_position, cka_value, upper_value in zip(
            x_positions,
            cka_values,
            ci_upper,
    ):
        text_y = min(
            upper_value + label_offset,
            zoom_upper * 0.96,
        )

        axis.text(
            x_position,
            text_y,
            f"{cka_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        data["Group"].astype(str),
        rotation=0,
    )

    axis.set_ylim(0.0, zoom_upper)
    axis.set_xlabel("")
    axis.set_ylabel("Linear CKA")
    # axis.set_title(
    #     "Cross-Modal Representational Similarity\n"
    #     "between BrainANet-SwinViT and RSFC"
    # )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    axis.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    # # 添加全尺度插图，防止局部放大造成视觉误导
    # inset_axis = inset_axes(
    #     axis,
    #     width="27%",
    #     height="40%",
    #     loc="center right",
    #     borderpad=1.5,
    # )
    #
    # inset_axis.errorbar(
    #     x_positions,
    #     cka_values,
    #     yerr=error_values,
    #     fmt="o",
    #     markersize=4,
    #     capsize=2,
    #     elinewidth=0.8,
    # )
    #
    # inset_axis.axhline(
    #     mean_cka,
    #     linestyle="--",
    #     linewidth=0.9,
    # )
    #
    # inset_axis.set_ylim(0.0, 1.0)
    # inset_axis.set_xticks([])
    # inset_axis.set_yticks([0.0, 0.5, 1.0])
    # inset_axis.set_title(
    #     "Full CKA scale",
    #     fontsize=8,
    # )
    # inset_axis.tick_params(labelsize=7)
    # inset_axis.grid(
    #     axis="y",
    #     linestyle="--",
    #     alpha=0.25,
    # )

    figure.tight_layout()

    figure.savefig(
        output_dir / "cka_by_group_optimized.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "cka_by_group_optimized.eps",
        bbox_inches="tight",
    )
    # figure.savefig(
    #     output_dir / "cka_by_group_optimized.pdf",
    #     bbox_inches="tight",
    # )

    plt.close(figure)

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain_df = read_table(args.brain)
    rsfc_df = read_table(args.rsfc)
    validate_keys(brain_df, args.id_col, args.group_col, "BrainANet file")
    validate_keys(rsfc_df, args.id_col, args.group_col, "RSFC file")

    excluded = parse_columns(args.exclude_cols)
    brain_features = select_feature_columns(
        brain_df, args.id_col, args.group_col, args.label_col,
        args.brain_feature_prefix, excluded,
    )
    rsfc_features = select_feature_columns(
        rsfc_df, args.id_col, args.group_col, args.label_col,
        args.rsfc_feature_prefix, excluded,
    )

    if args.group_col:
        groups = sorted(
            set(brain_df[args.group_col].dropna()) & set(rsfc_df[args.group_col].dropna()),
            key=str,
        )
    else:
        groups = ["All_subjects"]
    if not groups:
        raise ValueError("No common runs/folds were found.")

    rng = np.random.default_rng(args.random_seed)
    result_rows, bootstrap_rows, permutation_rows = [], [], []

    for group in groups:
        x, y, matched_ids = align_group(
            brain_df, rsfc_df, brain_features, rsfc_features,
            args.id_col, args.group_col, group, args.zscore,
        )
        observed = linear_cka(x, y)
        boot = bootstrap_cka(x, y, args.bootstrap, rng)
        ci_low, ci_high = percentile_ci(boot, args.confidence_level)

        if args.permutations > 0:
            null = permutation_cka(x, y, args.permutations, rng)
            p_value = float((1 + np.sum(null >= observed)) / (args.permutations + 1))
        else:
            null = np.asarray([], dtype=float)
            p_value = np.nan

        result_rows.append({
            "Group": group,
            "N_Matched_Subjects": x.shape[0],
            "BrainANet_Dimension": x.shape[1],
            "RSFC_Dimension": y.shape[1],
            "CKA": observed,
            "Bootstrap_CI_Lower": ci_low,
            "Bootstrap_CI_Upper": ci_high,
            "Permutation_P_Value": p_value,
            "Z_Scored": args.zscore,
        })
        bootstrap_rows.extend(
            {"Group": group, "Bootstrap_Index": i + 1, "CKA": value}
            for i, value in enumerate(boot)
        )
        permutation_rows.extend(
            {"Group": group, "Permutation_Index": i + 1, "Null_CKA": value}
            for i, value in enumerate(null)
        )
        pd.DataFrame({args.id_col: matched_ids}).to_csv(
            output_dir / f"matched_subjects_{group}.csv", index=False, encoding="utf-8-sig"
        )

    per_group = pd.DataFrame(result_rows)
    boot_df = pd.DataFrame(bootstrap_rows)
    perm_df = pd.DataFrame(permutation_rows)

    values = per_group["CKA"].to_numpy(float)
    mean_ci_low, mean_ci_high = aggregate_t_ci(values, args.confidence_level)
    summary = pd.DataFrame([{
        "Number_of_Groups": len(values),
        "Mean_CKA_Across_Groups": float(np.mean(values)),
        "SD_CKA_Across_Groups": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
        "CI_Lower_Across_Groups": mean_ci_low,
        "CI_Upper_Across_Groups": mean_ci_high,
        "Minimum_CKA": float(np.min(values)),
        "Maximum_CKA": float(np.max(values)),
    }])

    per_group.to_csv(output_dir / "cka_per_group.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    summary.to_csv(output_dir / "cka_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    boot_df.to_csv(output_dir / "cka_bootstrap_values.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    if not perm_df.empty:
        perm_df.to_csv(output_dir / "cka_permutation_null.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    with pd.ExcelWriter(output_dir / "cka_results.xlsx", engine="openpyxl") as writer:
        per_group.to_excel(writer, sheet_name="Per_Group", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        boot_df.to_excel(writer, sheet_name="Bootstrap", index=False)
        if not perm_df.empty:
            perm_df.to_excel(writer, sheet_name="Permutation_Null", index=False)

    save_plot(per_group, output_dir, args.confidence_level)

    settings = vars(args).copy()
    settings["brain_features"] = brain_features
    settings["rsfc_features"] = rsfc_features
    with open(output_dir / "cka_settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    print("\nCKA results by group:")
    print(per_group.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear CKA analysis for BrainANet and RSFC embeddings.")
    parser.add_argument("--brain", help="BrainANet embedding file.", default=r'cka_data/MDD.0_1/BrainANet.csv')
    parser.add_argument("--rsfc", help="RSFC embedding file.", default=r'cka_data/MDD.0_1/rsfc.csv')
    parser.add_argument("--id-col", default="Subject ID")
    parser.add_argument("--group-col", default='Run', help="Optional Run/Seed/Fold column.")
    parser.add_argument("--label-col", default="Group")
    parser.add_argument("--brain-feature-prefix", default=None)
    parser.add_argument("--rsfc-feature-prefix", default=None)
    parser.add_argument("--exclude-cols", default="")
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="cka_results/MDD.0_1")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must be between 0 and 1.")
    if args.bootstrap < 100:
        warnings.warn("Fewer than 100 bootstrap samples may produce unstable intervals.")
    if args.permutations < 0:
        raise ValueError("--permutations cannot be negative.")
    run(args)
