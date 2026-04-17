#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd

SEED = 42
FULL_PATH = "training_data_michal_full.csv"
TRAIN_PATH = "training_data_train.csv"
VAL_PATH = "training_data_val.csv"


def get_train_ratio(default: float = 0.78) -> float:
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(VAL_PATH)):
        return default
    n_train = len(pd.read_csv(TRAIN_PATH))
    n_val = len(pd.read_csv(VAL_PATH))
    total = n_train + n_val
    if total <= 0:
        return default
    return n_train / total


def main() -> None:
    if not os.path.exists(FULL_PATH):
        raise FileNotFoundError(f"Missing input file: {FULL_PATH}")

    ratio = get_train_ratio()
    df = pd.read_csv(FULL_PATH)

    train_parts = []
    val_parts = []

    for ex, grp in df.groupby("exercise", sort=False):
        g = grp.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(g)
        if n == 1:
            n_train = 1
        else:
            n_train = int(round(n * ratio))
            n_train = max(1, min(n - 1, n_train))

        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)

    n_train = len(train_df)
    n_val = len(val_df)
    ex_train = train_df["exercise"].nunique()
    ex_val = val_df["exercise"].nunique()

    print(f"ratio_used={ratio:.6f}")
    print(f"train_rows={n_train}")
    print(f"val_rows={n_val}")
    print(f"train_exercises={ex_train}")
    print(f"val_exercises={ex_val}")


if __name__ == "__main__":
    main()
