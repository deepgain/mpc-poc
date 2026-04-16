from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def resolve_exercise_col(df: pd.DataFrame) -> str:
    if "exercise_id" in df.columns:
        return "exercise_id"
    if "exercise" in df.columns:
        return "exercise"
    raise ValueError("Expected either 'exercise_id' or 'exercise' column")


def dist_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in ["rir", "reps", "weight_kg"]:
        q = df[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        out[col] = {
            "mean": float(df[col].mean()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "q05": float(q.loc[0.05]),
            "q25": float(q.loc[0.25]),
            "q50": float(q.loc[0.5]),
            "q75": float(q.loc[0.75]),
            "q95": float(q.loc[0.95]),
        }
    return out


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def write_report(
    out_path: Path,
    train_main: pd.DataFrame,
    val_main: pd.DataFrame,
    train_seed2: pd.DataFrame,
    val_seed2: pd.DataFrame,
) -> None:
    exercise_col = resolve_exercise_col(train_main)

    train_users = set(train_main["user_id"].unique())
    val_users = set(val_main["user_id"].unique())
    user_leakage = len(train_users.intersection(val_users))
    all_users = len(train_users.union(val_users))
    val_user_ratio = len(val_users) / all_users if all_users else 0.0

    train_ex = train_main.groupby(exercise_col).size().rename("train_sets")
    val_ex = val_main.groupby(exercise_col).size().rename("val_sets")
    cov = pd.concat([train_ex, val_ex], axis=1).fillna(0).astype(int)
    cov["total_sets"] = cov["train_sets"] + cov["val_sets"]
    missing_train = int((cov["train_sets"] == 0).sum())
    missing_val = int((cov["val_sets"] == 0).sum())
    min_train_sets = int(cov["train_sets"].min())
    min_val_sets = int(cov["val_sets"].min())

    main_dist = dist_stats(train_main)
    seed2_dist = dist_stats(train_seed2)

    summary = pd.DataFrame(
        {
            "metric": [
                "train_rows",
                "val_rows",
                "train_users",
                "val_users",
                "unique_exercises_train",
                "rir_mean_train",
                "reps_mean_train",
                "weight_mean_train",
            ],
            "baseline_main": [
                len(train_main),
                len(val_main),
                len(train_users),
                len(val_users),
                train_main[exercise_col].nunique(),
                train_main["rir"].mean(),
                train_main["reps"].mean(),
                train_main["weight_kg"].mean(),
            ],
            "baseline_seed2": [
                len(train_seed2),
                len(val_seed2),
                train_seed2["user_id"].nunique(),
                val_seed2["user_id"].nunique(),
                train_seed2[exercise_col].nunique(),
                train_seed2["rir"].mean(),
                train_seed2["reps"].mean(),
                train_seed2["weight_kg"].mean(),
            ],
        }
    )
    summary["abs_diff"] = (summary["baseline_main"] - summary["baseline_seed2"]).abs()
    summary["rel_diff_pct"] = (
        summary["abs_diff"] / summary[["baseline_main", "baseline_seed2"]].mean(axis=1) * 100
    )

    weight_zero_rows = int((train_main["weight_kg"] == 0).sum())
    weight_zero_pct = float((train_main["weight_kg"] == 0).mean() * 100)
    reps_20plus_rows = int((train_main["reps"] >= 20).sum())
    reps_20plus_pct = float((train_main["reps"] >= 20).mean() * 100)

    lines: list[str] = []
    lines.append("# Quality Validation Summary")
    lines.append("")
    lines.append("## 1. Split (baseline_main)")
    lines.append(f"- Train users: {len(train_users):,}")
    lines.append(f"- Val users: {len(val_users):,}")
    lines.append(f"- Total users: {all_users:,}")
    lines.append(f"- Val user ratio: {val_user_ratio*100:.2f}%")
    lines.append(f"- Leakage (intersection train/val users): {user_leakage}")
    lines.append("")
    lines.append("## 2. Coverage (baseline_main)")
    lines.append(f"- Unique exercises covered: {cov.shape[0]}")
    lines.append(f"- Exercises missing in train: {missing_train}")
    lines.append(f"- Exercises missing in val: {missing_val}")
    lines.append(f"- Minimum sets per exercise in train: {min_train_sets}")
    lines.append(f"- Minimum sets per exercise in val: {min_val_sets}")
    lines.append("")
    lines.append("## 3. Distribution realism (train only)")
    for name, stats in [("baseline_main", main_dist), ("baseline_seed2", seed2_dist)]:
        lines.append(f"### {name}")
        for col in ["rir", "reps", "weight_kg"]:
            st = stats[col]
            lines.append(
                f"- {col}: mean={st['mean']:.3f}, min/max={st['min']:.3f}/{st['max']:.3f}, "
                f"q05={st['q05']:.3f}, q25={st['q25']:.3f}, q50={st['q50']:.3f}, q75={st['q75']:.3f}, q95={st['q95']:.3f}"
            )
        lines.append("")

    lines.append("### Outlier sanity")
    lines.append(f"- Rows with weight_kg == 0 (train): {weight_zero_rows} ({weight_zero_pct:.3f}%)")
    lines.append(f"- Rows with reps >= 20 (train): {reps_20plus_rows} ({reps_20plus_pct:.3f}%)")
    lines.append("")

    lines.append("## 4. Seed stability (baseline_main vs baseline_seed2)")
    lines.append("| metric | baseline_main | baseline_seed2 | abs_diff | rel_diff_pct |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['metric']} | {row['baseline_main']:.3f} | {row['baseline_seed2']:.3f} | {row['abs_diff']:.3f} | {row['rel_diff_pct']:.3f} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote report: {out_path}")
    print(f"Leakage users: {user_leakage}")
    print(f"Val user ratio: {val_user_ratio*100:.2f}%")
    print(f"Coverage missing train/val: {missing_train}/{missing_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated dataset quality")
    parser.add_argument(
        "--root",
        default="generated_datasets",
        help="Root directory containing baseline_main and baseline_seed2 folders",
    )
    parser.add_argument(
        "--output",
        default="generated_datasets/quality_validation.md",
        help="Output markdown report path",
    )
    args = parser.parse_args()

    root = Path(args.root)
    train_main = load_csv(root / "baseline_main" / "training_data_train.csv")
    val_main = load_csv(root / "baseline_main" / "training_data_val.csv")
    train_seed2 = load_csv(root / "baseline_seed2" / "training_data_train.csv")
    val_seed2 = load_csv(root / "baseline_seed2" / "training_data_val.csv")

    write_report(Path(args.output), train_main, val_main, train_seed2, val_seed2)


if __name__ == "__main__":
    main()
