#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inference import EXERCISE_TO_IDX, RIR_SCALE, load_model
from strength_priors import DEFAULT_SESSION_GAP_HOURS
from validate_1rm_dynamics import build_anchor_histories, sequential_predict


DEFAULT_CHECKPOINT = "deepgain_model_best.pt"
DEFAULT_DATA_PATH = "training_data.csv"
DEFAULT_OUTPUT_PATH = os.path.join("charts", "chart_dynamic_vs_static_rir.png")
PINNED_USER_IDS = {"user_00030", "user_00111"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dynamic-vs-static RIR chart without retraining.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_holdout_df(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["exercise_idx"] = df["exercise"].map(EXERCISE_TO_IDX)
    df = df.dropna(subset=["exercise_idx"]).copy()
    df["exercise_idx"] = df["exercise_idx"].astype(int)

    user_ids = df["user_id"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(user_ids)
    split = int(0.7 * len(user_ids))
    holdout_users = set(user_ids[split:])
    return df[df["user_id"].isin(holdout_users)].copy()


def select_sample_users(holdout_df: pd.DataFrame) -> list[str]:
    groups = []
    for user_id, grp in holdout_df.groupby("user_id"):
        groups.append((user_id, len(grp)))

    pinned = [user_id for user_id, _ in groups if user_id in PINNED_USER_IDS]
    rest = [(user_id, n_sets) for user_id, n_sets in groups if user_id not in PINNED_USER_IDS]
    rest_sorted = [user_id for user_id, _ in sorted(rest, key=lambda item: item[1], reverse=True)]

    chosen = (pinned + rest_sorted)[:3]
    return chosen


def compute_session_starts(timestamps: pd.Series, gap_hours: float = DEFAULT_SESSION_GAP_HOURS) -> np.ndarray:
    deltas_h = timestamps.diff().dt.total_seconds().div(3600.0).fillna(0.0)
    starts = np.flatnonzero(deltas_h.to_numpy() > gap_hours)
    return starts.astype(int)


def main() -> int:
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = load_model(args.checkpoint)
    holdout_df = load_holdout_df(args.data)
    sample_user_ids = select_sample_users(holdout_df)

    fig, axes = plt.subplots(len(sample_user_ids), 1, figsize=(16, 4.2 * len(sample_user_ids)), sharex=False)
    if len(sample_user_ids) == 1:
        axes = [axes]

    for ax, user_id in zip(axes, sample_user_ids):
        grp = holdout_df[holdout_df["user_id"] == user_id].sort_values("timestamp").reset_index(drop=True)
        dynamic_history, static_history, _, _ = build_anchor_histories(grp)
        rir_dyn = sequential_predict(model, grp, dynamic_history)
        rir_static = sequential_predict(model, grp, static_history)
        rir_actual = grp["rir"].to_numpy(dtype=np.float32)

        T_plot = min(len(grp), 250)
        grp = grp.iloc[:T_plot].reset_index(drop=True)
        rir_dyn = rir_dyn[:T_plot]
        rir_static = rir_static[:T_plot]
        rir_actual = rir_actual[:T_plot]
        x = np.arange(T_plot, dtype=np.int32)
        session_starts = compute_session_starts(grp["timestamp"])

        dyn_mae = float(np.mean(np.abs(rir_dyn - rir_actual)))
        static_mae = float(np.mean(np.abs(rir_static - rir_actual)))

        for start in session_starts:
            ax.axvline(start - 0.5, color="#bbbbbb", linestyle=":", linewidth=0.9, alpha=0.9, zorder=0)

        ax.scatter(x, rir_actual, color="black", s=16, alpha=0.85, label="actual RIR", zorder=3)
        ax.plot(x, rir_dyn, color="#1f77b4", linewidth=1.8, label=f"dynamic anchors (MAE={dyn_mae:.2f})", zorder=2)
        ax.plot(x, rir_static, color="#ff7f0e", linewidth=1.6, linestyle="--", label=f"static onboarding (MAE={static_mae:.2f})", zorder=2)
        ax.set_title(
            f"{user_id} — Dynamic vs static anchors by set "
            f"(sets={T_plot}, sessions={len(session_starts) + 1})"
        )
        ax.set_ylabel("RIR")
        ax.set_ylim(-0.1, 5.1)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Set index")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
