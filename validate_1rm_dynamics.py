from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from inference import (
    ALL_MUSCLES,
    DT_SCALE,
    EXERCISE_TO_IDX,
    REPS_SCALE,
    RIR_SCALE,
    WEIGHT_SCALE,
    load_model,
)
from strength_priors import ANCHOR_NAMES, resolve_anchor_values, update_strength_anchors


DEFAULT_CHECKPOINT = "deepgain_model_best.pt"
DEFAULT_DATA_PATH = "training_data.csv"
DEFAULT_NUM_USERS = 12
DEFAULT_SPLIT_SEED = 42
DEFAULT_USER_SAMPLE_SEED = 123


@dataclass
class EvalResult:
    rmse: float
    mae: float
    corr: float
    n_sets: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dynamic 1RM anchor behavior on held-out users.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--num-users", type=int, default=DEFAULT_NUM_USERS)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--user-sample-seed", type=int, default=DEFAULT_USER_SAMPLE_SEED)
    return parser.parse_args()


def load_eval_users(data_path: str, num_users: int, split_seed: int, sample_seed: int) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["exercise_idx"] = df["exercise"].map(EXERCISE_TO_IDX)
    df = df.dropna(subset=["exercise_idx"]).copy()
    df["exercise_idx"] = df["exercise_idx"].astype(int)

    user_ids = df["user_id"].unique()
    split_rng = np.random.RandomState(split_seed)
    split_rng.shuffle(user_ids)
    split = int(0.7 * len(user_ids))
    val_users = np.array(user_ids[split:])

    sample_rng = np.random.RandomState(sample_seed)
    if num_users < len(val_users):
        chosen_users = sample_rng.choice(val_users, size=num_users, replace=False)
    else:
        chosen_users = val_users

    out = df[df["user_id"].isin(set(chosen_users))].copy()
    out = out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return out


def row_to_completed_set(row) -> dict:
    return {
        "exercise": row.exercise,
        "weight_kg": float(row.weight_kg),
        "reps": int(row.reps),
        "rir": float(row.rir),
        "timestamp": row.timestamp,
    }


def build_anchor_histories(user_df: pd.DataFrame, override_initial_anchors=None) -> tuple[np.ndarray, np.ndarray]:
    initial_anchors = resolve_anchor_values(
        anchor_values=override_initial_anchors if override_initial_anchors is not None else user_df.iloc[0].to_dict()
    )
    current_anchors = initial_anchors.copy()
    dynamic_history = []
    static_history = []

    for row in user_df.itertuples(index=False):
        dynamic_history.append(current_anchors.copy())
        static_history.append(initial_anchors.copy())
        current_anchors = update_strength_anchors(current_anchors, [row_to_completed_set(row)])

    return np.stack(dynamic_history, axis=0), np.stack(static_history, axis=0)


def sequential_predict(model, user_df: pd.DataFrame, anchor_history_kg: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    num_muscles = len(ALL_MUSCLES)

    all_m_idx = torch.arange(num_muscles, device=device)
    all_m_embed = model.muscle_embed(all_m_idx)

    preds = []
    prev_ts = None
    mpc = torch.ones(1, num_muscles, device=device)

    with torch.no_grad():
        for row, anchors_kg in zip(user_df.itertuples(index=False), anchor_history_kg):
            if prev_ts is not None:
                dt_h = (row.timestamp - prev_ts).total_seconds() / 3600.0
                if dt_h > 0.0:
                    dt_norm = torch.tensor(
                        [np.log1p(dt_h) / DT_SCALE], dtype=torch.float32, device=device
                    ).expand(num_muscles)
                    mpc = model.r(mpc.reshape(-1), dt_norm, all_m_idx).reshape(1, num_muscles)

            ei = torch.tensor([row.exercise_idx], dtype=torch.long, device=device)
            w_t = torch.tensor([float(row.weight_kg) / WEIGHT_SCALE], dtype=torch.float32, device=device)
            r_t = torch.tensor([float(row.reps) / REPS_SCALE], dtype=torch.float32, device=device)
            rir_t = torch.tensor([float(row.rir) / RIR_SCALE], dtype=torch.float32, device=device)
            anchors_t = torch.tensor([anchors_kg / WEIGHT_SCALE], dtype=torch.float32, device=device)

            rir_pred = model.predict_rir_norm(ei, w_t, r_t, mpc, anchors_t).item() * RIR_SCALE
            preds.append(float(rir_pred))

            inv = model.involvement[ei]
            drop = model.predict_drop_norm(
                ei.expand(num_muscles),
                w_t.expand(num_muscles),
                r_t.expand(num_muscles),
                rir_t.expand(num_muscles),
                mpc.reshape(-1),
                all_m_embed,
                anchors_t.expand(num_muscles, -1),
            ).reshape(1, num_muscles)
            mpc = (mpc * (1.0 - inv * drop)).clamp(min=0.1)
            prev_ts = row.timestamp

    return np.asarray(preds, dtype=np.float32)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> EvalResult:
    mse = float(np.mean((preds - targets) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(preds - targets)))
    corr = float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else float("nan")
    return EvalResult(rmse=rmse, mae=mae, corr=corr, n_sets=len(preds))


def build_shuffled_anchor_map(eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    user_groups = list(eval_df.groupby("user_id"))
    initial_anchors = [
        resolve_anchor_values(anchor_values=grp.iloc[0].to_dict())
        for _, grp in user_groups
    ]
    shuffled = {}
    for idx, (user_id, _) in enumerate(user_groups):
        shuffled[user_id] = initial_anchors[(idx + 1) % len(initial_anchors)].copy()
    return shuffled


def summarize_anchor_dynamics(dynamic_histories: list[np.ndarray], static_histories: list[np.ndarray]) -> dict[str, float]:
    changed_users = 0
    total_sets = 0
    changed_sets = 0
    final_relative_changes = {anchor: [] for anchor in ANCHOR_NAMES}

    for dynamic_history, static_history in zip(dynamic_histories, static_histories):
        diff = np.abs(dynamic_history - static_history)
        user_changed = bool(np.any(diff > 1e-9))
        changed_users += int(user_changed)
        changed_set_mask = np.any(diff > 1e-9, axis=1)
        changed_sets += int(changed_set_mask.sum())
        total_sets += dynamic_history.shape[0]

        initial = static_history[0]
        final = dynamic_history[-1]
        for idx, anchor in enumerate(ANCHOR_NAMES):
            denom = max(float(initial[idx]), 1e-6)
            final_relative_changes[anchor].append(float((final[idx] - initial[idx]) / denom))

    return {
        "changed_user_fraction": changed_users / max(len(dynamic_histories), 1),
        "changed_set_fraction": changed_sets / max(total_sets, 1),
        **{
            f"{anchor}_median_final_rel_change": float(np.median(values))
            for anchor, values in final_relative_changes.items()
        },
    }


def main() -> int:
    args = parse_args()
    model = load_model(args.checkpoint)
    eval_df = load_eval_users(args.data, args.num_users, args.split_seed, args.user_sample_seed)

    if model.strength_feature_dim <= 0:
        print("FAIL loaded checkpoint without strength features")
        return 1

    print(f"checkpoint = {args.checkpoint}")
    print(f"data = {args.data}")
    print(f"eval_users = {eval_df['user_id'].nunique()}  eval_sets = {len(eval_df):,}")
    print()

    shuffle_map = build_shuffled_anchor_map(eval_df)

    preds_dynamic = []
    preds_static = []
    preds_shuffled = []
    targets = []
    dynamic_histories = []
    static_histories = []

    for user_id, grp in eval_df.groupby("user_id"):
        dynamic_history, static_history = build_anchor_histories(grp)
        shuffled_dynamic_history, _ = build_anchor_histories(grp, override_initial_anchors=shuffle_map[user_id])

        dynamic_histories.append(dynamic_history)
        static_histories.append(static_history)

        preds_dynamic.append(sequential_predict(model, grp, dynamic_history))
        preds_static.append(sequential_predict(model, grp, static_history))
        preds_shuffled.append(sequential_predict(model, grp, shuffled_dynamic_history))
        targets.append(grp["rir"].to_numpy(dtype=np.float32))

    preds_dynamic = np.concatenate(preds_dynamic)
    preds_static = np.concatenate(preds_static)
    preds_shuffled = np.concatenate(preds_shuffled)
    targets = np.concatenate(targets)

    dynamic_metrics = compute_metrics(preds_dynamic, targets)
    static_metrics = compute_metrics(preds_static, targets)
    shuffled_metrics = compute_metrics(preds_shuffled, targets)
    anchor_stats = summarize_anchor_dynamics(dynamic_histories, static_histories)

    print("== Holdout Metrics ==")
    print(
        f"dynamic_correct   rmse={dynamic_metrics.rmse:.4f} "
        f"mae={dynamic_metrics.mae:.4f} corr={dynamic_metrics.corr:.4f}"
    )
    print(
        f"static_onboarding rmse={static_metrics.rmse:.4f} "
        f"mae={static_metrics.mae:.4f} corr={static_metrics.corr:.4f}"
    )
    print(
        f"shuffled_dynamic  rmse={shuffled_metrics.rmse:.4f} "
        f"mae={shuffled_metrics.mae:.4f} corr={shuffled_metrics.corr:.4f}"
    )
    print()

    print("== Dynamic vs Static Gain ==")
    print(f"rmse_gain = {static_metrics.rmse - dynamic_metrics.rmse:.4f}")
    print(f"mae_gain  = {static_metrics.mae - dynamic_metrics.mae:.4f}")
    print()

    print("== Dynamic vs Shuffled Gain ==")
    print(f"rmse_gain = {shuffled_metrics.rmse - dynamic_metrics.rmse:.4f}")
    print(f"mae_gain  = {shuffled_metrics.mae - dynamic_metrics.mae:.4f}")
    print()

    print("== Anchor Dynamics Stats ==")
    print(f"changed_user_fraction = {anchor_stats['changed_user_fraction']:.3f}")
    print(f"changed_set_fraction  = {anchor_stats['changed_set_fraction']:.3f}")
    for anchor in ANCHOR_NAMES:
        print(f"{anchor}_median_final_rel_change = {anchor_stats[f'{anchor}_median_final_rel_change']:.4f}")
    print()

    pass_dynamic_beats_static = dynamic_metrics.rmse <= static_metrics.rmse + 1e-6
    pass_dynamic_beats_shuffled = dynamic_metrics.rmse <= shuffled_metrics.rmse + 1e-6
    pass_dynamic_changes = anchor_stats["changed_user_fraction"] > 0.0 and anchor_stats["changed_set_fraction"] > 0.0

    print("== Verdict ==")
    print(f"dynamic_better_than_static   = {pass_dynamic_beats_static}")
    print(f"dynamic_better_than_shuffled = {pass_dynamic_beats_shuffled}")
    print(f"anchors_change_over_time     = {pass_dynamic_changes}")

    overall_pass = pass_dynamic_beats_static and pass_dynamic_beats_shuffled and pass_dynamic_changes
    print()
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
