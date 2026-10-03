#!/usr/bin/env python3
"""
train.py — Model training for the entity resolution pipeline.

Builds labeled training data from ground truth pairs and random negatives,
trains an XGBoost classifier, and optimises the decision threshold
for macro-averaged F_0.5 (the competition metric).
"""

import gc
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from preprocessing import load_source_chunked
from features import FEATURE_COLS, compute_features


def load_ground_truth(gt_path):
    """Parse ground truth TSV into a set of (s1_id, matched_id) pairs."""
    pairs = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0].strip()
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        pairs.add((s1_id, mid))
    return pairs


def sample_training_pairs(gt_pairs, s1_lookup, other_lookup,
                          sample_size=300000, neg_ratio=1.0, seed=42):
    """
    Sample positive and negative training pairs.
    
    Positives are drawn from ground truth. Negatives are randomly
    paired from the same country to ensure country_agreement feature
    doesn't trivially separate classes.
    """
    rng = np.random.RandomState(seed)

    # Filter to pairs where both entities exist in our lookup tables
    valid_pos = [(s1, oth) for s1, oth in gt_pairs
                 if s1 in s1_lookup.index and oth in other_lookup.index]
    print(f"  Valid positive pairs: {len(valid_pos):,} / {len(gt_pairs):,}")

    if len(valid_pos) > sample_size:
        idx = rng.choice(len(valid_pos), sample_size, replace=False)
        valid_pos = [valid_pos[i] for i in idx]
    print(f"  Sampled positives: {len(valid_pos):,}")

    # Group other entity IDs by country for fast same-country negative sampling
    country_series = other_lookup["norm_country"]
    other_by_country = {c: idx.to_numpy() for c, idx in country_series.groupby(country_series).groups.items()}
    s1_country_map = s1_lookup["norm_country"].to_dict()

    n_neg_target = int(len(valid_pos) * neg_ratio)
    neg_rng = np.random.RandomState(seed + 1)
    neg_list = []

    for s1_id, _ in valid_pos:
        if len(neg_list) >= n_neg_target:
            break
        country = s1_country_map.get(s1_id, "")
        pool = other_by_country.get(country, None)
        if pool is None or len(pool) == 0:
            continue
        for _ in range(5):
            candidate = pool[neg_rng.randint(len(pool))]
            if (s1_id, candidate) not in gt_pairs:
                neg_list.append((s1_id, candidate))
                break

    print(f"  Negatives generated: {len(neg_list):,}")
    return valid_pos, neg_list


def train_model(train_s1_path, train_s2_path, train_s3_path, gt_path,
                sample_size=300000, nrows=None):
    """
    Full training pipeline: load data, compute features, train XGBoost.
    
    Returns (model, threshold, feature_cols).
    """
    from xgboost import XGBClassifier

    t0 = time.time()

    # Load and preprocess sources
    print("Loading training sources ...")
    s1 = load_source_chunked(train_s1_path, nrows=nrows)
    s2 = load_source_chunked(train_s2_path, nrows=nrows)
    s3 = load_source_chunked(train_s3_path, nrows=nrows)

    # Build indexed lookups
    s1_lookup = s1.set_index("entity_id")[
        ["norm_name", "norm_address", "norm_country"]
    ]
    other_lookup = pd.concat([
        s2[["entity_id", "norm_name", "norm_address", "norm_country"]],
        s3[["entity_id", "norm_name", "norm_address", "norm_country"]],
    ], ignore_index=True).set_index("entity_id")
    del s1, s2, s3
    gc.collect()

    print(f"Lookups: S1={len(s1_lookup):,}, S2+S3={len(other_lookup):,}")

    # Load ground truth and sample pairs
    gt_pairs = load_ground_truth(gt_path)
    print(f"Ground truth pairs: {len(gt_pairs):,}")

    pos_pairs, neg_pairs = sample_training_pairs(
        gt_pairs, s1_lookup, other_lookup, sample_size=sample_size
    )

    # Compute features
    print("Computing features for positives ...")
    pos_df = pd.DataFrame(pos_pairs,
                          columns=["source1_entity_id", "matched_entity_id"])
    pos_df = compute_features(pos_df, s1_lookup, other_lookup)
    pos_df["label"] = 1

    print("Computing features for negatives ...")
    neg_df = pd.DataFrame(neg_pairs,
                          columns=["source1_entity_id", "matched_entity_id"])
    neg_df = compute_features(neg_df, s1_lookup, other_lookup)
    neg_df["label"] = 0

    train_df = pd.concat([pos_df, neg_df], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Training set: {len(train_df):,} samples "
          f"(pos={len(pos_df):,}, neg={len(neg_df):,})")

    del pos_df, neg_df, s1_lookup, other_lookup
    gc.collect()

    # Train/val split
    n_val = int(len(train_df) * 0.15)
    val_df = train_df.iloc[:n_val]
    trn_df = train_df.iloc[n_val:]

    X_train = trn_df[FEATURE_COLS].values.astype(np.float32)
    y_train = trn_df["label"].values.astype(np.int32)
    X_val = val_df[FEATURE_COLS].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int32)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / max(pos_count, 1)

    print(f"Train split: {len(trn_df):,} (pos={pos_count:,}, neg={neg_count:,})")
    print(f"Val split: {len(val_df):,}")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    print("Training XGBoost ...")
    model.fit(X_train, y_train)

    # Optimise threshold for F_0.5
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thr, best_f05 = 0.5, 0.0

    for thr_pct in range(30, 96):
        thr = thr_pct / 100.0
        preds = (val_probs >= thr).astype(int)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = 0.25 * prec + rec
        f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0

        if f05 > best_f05:
            best_f05 = f05
            best_thr = thr

    print(f"Optimal threshold: {best_thr:.2f} (Val F_0.5: {best_f05:.4f})")

    # Report feature importances
    imp = model.feature_importances_
    fi = sorted(zip(FEATURE_COLS, imp), key=lambda x: x[1], reverse=True)
    print("Feature importances:")
    for fname, fval in fi:
        print(f"  {fname:<28} {fval:.4f}")

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    return model, best_thr
