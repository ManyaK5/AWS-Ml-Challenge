#!/usr/bin/env python3
"""
inference.py — Score candidate pairs and produce the final matching output.

Reads candidate_pairs.tsv, computes features in batches, applies the trained
XGBoost model, and writes matching_results.tsv.
"""

import gc
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from preprocessing import load_source_chunked
from features import FEATURE_COLS, compute_features


def run_inference(model, threshold, candidate_pairs_path,
                  test_s1_path, test_s2_path, test_s3_path,
                  output_path, batch_size=500_000):
    """
    Score all candidate pairs and write the matching results.
    
    Processes candidates in batches to control memory usage.
    Each batch:
      1. Reads a chunk of (s1_id, candidate_id) pairs
      2. Joins with source data to get text fields
      3. Computes similarity features
      4. Runs XGBoost inference
      5. Collects pairs above threshold
    """
    t0 = time.time()
    print(f"Inference starting (threshold={threshold:.2f})")

    # Load and preprocess test sources
    print("Loading test sources ...")
    test_s1 = load_source_chunked(test_s1_path)
    test_s2 = load_source_chunked(test_s2_path)
    test_s3 = load_source_chunked(test_s3_path)

    # Build lookups
    s1_lookup = test_s1.set_index("entity_id")[
        ["norm_name", "norm_address", "norm_country"]
    ]
    other_lookup = pd.concat([
        test_s2[["entity_id", "norm_name", "norm_address", "norm_country"]],
        test_s3[["entity_id", "norm_name", "norm_address", "norm_country"]],
    ], ignore_index=True).set_index("entity_id")

    all_s1_ids = sorted(test_s1["entity_id"].unique())
    del test_s1, test_s2, test_s3
    gc.collect()

    print(f"Lookups: S1={len(s1_lookup):,}, S2+S3={len(other_lookup):,}")

    # Stream and score candidates
    matched_mapping = defaultdict(list)
    total_scored = 0
    total_matched = 0
    batch_pairs = []

    print(f"Scoring candidates from {os.path.basename(candidate_pairs_path)} ...")

    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            if not cand_str:
                continue

            for cid in cand_str.split(","):
                cid = cid.strip()
                if cid:
                    batch_pairs.append((s1_id, cid))

            if len(batch_pairs) >= batch_size:
                n_m = _score_batch(batch_pairs, s1_lookup, other_lookup,
                                   model, threshold, matched_mapping)
                total_scored += len(batch_pairs)
                total_matched += n_m
                pct = total_scored / 54_000_000 * 100
                elapsed = time.time() - t0
                print(f"  {total_scored:>12,} scored | "
                      f"{total_matched:>8,} matched | "
                      f"{pct:.1f}% | {elapsed:.0f}s")
                batch_pairs = []

    # Final batch
    if batch_pairs:
        n_m = _score_batch(batch_pairs, s1_lookup, other_lookup,
                           model, threshold, matched_mapping)
        total_scored += len(batch_pairs)
        total_matched += n_m

    elapsed = time.time() - t0
    print(f"\nScoring complete: {total_scored:,} pairs, "
          f"{total_matched:,} matches in {elapsed:.1f}s")
    print(f"Entities with matches: {len(matched_mapping):,}")

    # Write output
    write_matching_results(all_s1_ids, matched_mapping, output_path)
    return matched_mapping


def _score_batch(batch_pairs, s1_lookup, other_lookup,
                 model, threshold, mapping):
    """Score one batch and update the mapping dict."""
    batch_df = pd.DataFrame(batch_pairs,
                            columns=["source1_entity_id", "matched_entity_id"])
    batch_df = compute_features(batch_df, s1_lookup, other_lookup)
    batch_df = batch_df.dropna(subset=FEATURE_COLS)

    if batch_df.empty:
        return 0

    # Score candidates using hybrid matcher or base estimator
    try:
        probs = model.predict_proba(batch_df)
        if probs.ndim == 2:
            probs = probs[:, 1]
    except (TypeError, ValueError, KeyError):
        feats = getattr(model, "feature_cols", FEATURE_COLS)
        probs = model.predict_proba(
            batch_df[feats].values.astype(np.float32)
        )[:, 1]

    mask = probs >= threshold
    hits = batch_df.loc[mask, ["source1_entity_id", "matched_entity_id"]]

    n_match = 0
    for s1_id, oth_id in zip(hits["source1_entity_id"],
                              hits["matched_entity_id"]):
        mapping[s1_id].append(oth_id)
        n_match += 1

    del batch_df
    return n_match


def write_matching_results(all_s1_ids, matched_mapping, output_path):
    """Write matching_results.tsv in the competition format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_matched = 0
    n_singleton = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = matched_mapping.get(s1_id, [])
            if matches:
                unique_matches = sorted(set(matches))
                f.write(f"{s1_id}\t{','.join(unique_matches)}\n")
                n_matched += 1
            else:
                f.write(f"{s1_id}\t\n")
                n_singleton += 1

    total = n_matched + n_singleton
    print(f"Written {output_path}: {total:,} rows "
          f"({n_matched:,} matched, {n_singleton:,} singletons)")
