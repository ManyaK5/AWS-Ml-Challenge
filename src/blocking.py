#!/usr/bin/env python3
"""
blocking.py — Multi-pass candidate generation for entity resolution.

Implements six blocking strategies that reduce the O(n*m) comparison space
to a tractable candidate set. Each strategy targets a different type of
record variation, and their union provides high recall coverage.

Strategies:
  A. Exact normalised name matching
  B. Exact address + country matching
  C. Rare-token blocking on business names
  D. Rare-token blocking on addresses (tight thresholds)
  E. Postal prefix + country blocking
  G. Sorted neighbourhood on normalised names
"""

import logging
from collections import defaultdict
from typing import Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PairSet = Set[Tuple[str, str]]


def block_by_exact_name(s1, s_other, max_block_size=30):
    """
    Candidate pairs where normalised business names match exactly.
    Blocks larger than max_block_size on either side are pruned
    to control quadratic blowup.
    """
    if s1.empty or s_other.empty:
        return set()

    s1_clean = s1[s1["norm_name"] != ""][["entity_id", "norm_name"]].copy()
    other_clean = s_other[s_other["norm_name"] != ""][["entity_id", "norm_name"]].copy()

    if s1_clean.empty or other_clean.empty:
        return set()

    s1_cnts = s1_clean.groupby("norm_name")["entity_id"].count().rename("n_s1")
    other_cnts = other_clean.groupby("norm_name")["entity_id"].count().rename("n_oth")

    valid_keys = (
        s1_cnts.to_frame()
        .join(other_cnts.to_frame(), how="inner")
        .query(f"n_s1 <= {max_block_size} and n_oth <= {max_block_size}")
        .index
    )
    if len(valid_keys) == 0:
        return set()

    s1_v = s1_clean[s1_clean["norm_name"].isin(valid_keys)]
    other_v = other_clean[other_clean["norm_name"].isin(valid_keys)]

    merged = s1_v.merge(other_v, on="norm_name", suffixes=("_s1", "_oth"))
    pairs = set(zip(merged["entity_id_s1"], merged["entity_id_oth"]))
    logger.info("Exact-name blocking: %d pairs", len(pairs))
    return pairs


def block_by_address_country(s1, s_other, max_block_size=30):
    """
    Candidate pairs sharing identical normalised address AND country.
    Catches businesses with different names at the same location.
    """
    if s1.empty or s_other.empty:
        return set()

    cols = ["entity_id", "norm_address", "norm_country"]
    s1_c = s1[s1["norm_address"] != ""][cols].copy()
    other_c = s_other[s_other["norm_address"] != ""][cols].copy()

    if s1_c.empty or other_c.empty:
        return set()

    key = ["norm_address", "norm_country"]
    s1_cnts = s1_c.groupby(key)["entity_id"].count().rename("n_s1")
    other_cnts = other_c.groupby(key)["entity_id"].count().rename("n_oth")

    valid_keys = (
        s1_cnts.to_frame()
        .join(other_cnts.to_frame(), how="inner")
        .query(f"n_s1 <= {max_block_size} and n_oth <= {max_block_size}")
        .reset_index()[key]
    )
    if valid_keys.empty:
        return set()

    s1_v = s1_c.merge(valid_keys, on=key)
    other_v = other_c.merge(valid_keys, on=key)

    merged = s1_v.merge(other_v, on=key, suffixes=("_s1", "_oth"))
    pairs = set(zip(merged["entity_id_s1"], merged["entity_id_oth"]))
    logger.info("Address+country blocking: %d pairs", len(pairs))
    return pairs


def block_by_rare_token(s1, s_other, field, denylist,
                        max_docs=50, max_block_size=50):
    """
    Pairs sharing a rare token in the specified field.
    A token is rare if it appears in <= max_docs distinct entities
    on each side. Tokens in denylist are always excluded.
    """
    if s1.empty or s_other.empty:
        return set()

    def tokenise_df(df):
        sub = df[df[field].notna() & (df[field] != "")][["entity_id", field]]
        exploded = sub.assign(token=sub[field].str.split()).explode("token")
        exploded["token"] = exploded["token"].str.strip()
        mask = (exploded["token"].str.len() >= 3) & ~exploded["token"].isin(denylist)
        return exploded.loc[mask, ["entity_id", "token"]]

    tok_s1 = tokenise_df(s1)
    tok_other = tokenise_df(s_other)

    if tok_s1.empty or tok_other.empty:
        return set()

    cnt_s1 = tok_s1.groupby("token")["entity_id"].nunique().rename("n_s1")
    cnt_other = tok_other.groupby("token")["entity_id"].nunique().rename("n_oth")

    valid_tokens = (
        cnt_s1.to_frame()
        .join(cnt_other.to_frame(), how="inner")
        .query(f"n_s1 <= {max_docs} and n_oth <= {max_docs}")
        .index
    )
    if len(valid_tokens) == 0:
        return set()

    tok_s1_f = tok_s1[tok_s1["token"].isin(valid_tokens)].drop_duplicates(
        ["entity_id", "token"]
    )
    tok_other_f = tok_other[tok_other["token"].isin(valid_tokens)].drop_duplicates(
        ["entity_id", "token"]
    )

    # Apply block size guard
    s1_cnt_per_tok = tok_s1_f.groupby("token")["entity_id"].count()
    oth_cnt_per_tok = tok_other_f.groupby("token")["entity_id"].count()
    ok_tokens = (
        s1_cnt_per_tok[s1_cnt_per_tok <= max_block_size].index
        .intersection(oth_cnt_per_tok[oth_cnt_per_tok <= max_block_size].index)
    )
    if len(ok_tokens) == 0:
        return set()

    tok_s1_f = tok_s1_f[tok_s1_f["token"].isin(ok_tokens)]
    tok_other_f = tok_other_f[tok_other_f["token"].isin(ok_tokens)]

    merged = tok_s1_f.merge(tok_other_f, on="token", suffixes=("_s1", "_oth"))
    pairs = set(zip(merged["entity_id_s1"], merged["entity_id_oth"]))
    logger.info("Rare-token blocking (field=%s): %d pairs", field, len(pairs))
    return pairs


def block_by_postal_prefix(s1, s_other, max_block_size=100):
    """
    Pairs sharing the same postal/ZIP prefix and country.
    Extracts the first 5-digit numeric token from normalised addresses.
    """
    if s1.empty or s_other.empty:
        return set()

    def extract_postal(addr):
        if not addr:
            return ""
        tokens = addr.split()
        for tok in reversed(tokens):
            if tok.isdigit() and len(tok) >= 4:
                return tok[:5]
        for tok in tokens:
            if tok.isdigit() and len(tok) >= 4:
                return tok[:5]
        return ""

    s1_c = s1[["entity_id", "norm_address", "norm_country"]].copy()
    s1_c["postal_prefix"] = s1_c["norm_address"].apply(extract_postal)
    other_c = s_other[["entity_id", "norm_address", "norm_country"]].copy()
    other_c["postal_prefix"] = other_c["norm_address"].apply(extract_postal)

    s1_c = s1_c[s1_c["postal_prefix"] != ""]
    other_c = other_c[other_c["postal_prefix"] != ""]

    if s1_c.empty or other_c.empty:
        return set()

    key = ["postal_prefix", "norm_country"]
    s1_cnts = s1_c.groupby(key)["entity_id"].count().rename("n_s1")
    other_cnts = other_c.groupby(key)["entity_id"].count().rename("n_oth")

    valid_keys = (
        s1_cnts.to_frame()
        .join(other_cnts.to_frame(), how="inner")
        .query(f"n_s1 <= {max_block_size} and n_oth <= {max_block_size}")
        .reset_index()[key]
    )
    if valid_keys.empty:
        return set()

    s1_v = s1_c.merge(valid_keys, on=key)
    other_v = other_c.merge(valid_keys, on=key)

    merged = s1_v.merge(other_v, on=key, suffixes=("_s1", "_oth"))
    pairs = set(zip(merged["entity_id_s1"], merged["entity_id_oth"]))
    logger.info("Postal-prefix blocking: %d pairs", len(pairs))
    return pairs


def block_by_sorted_neighbourhood(s1, s_other, field="norm_name", window=5):
    """
    Sort all records by a field, slide a window of size W,
    and emit cross-source pairs within each window position.
    """
    pairs = set()
    if s1.empty or s_other.empty:
        return pairs

    s1_sub = s1[["entity_id", field]].assign(src="s1")
    other_sub = s_other[["entity_id", field]].assign(src="oth")
    combined = pd.concat([s1_sub, other_sub], ignore_index=True)
    combined = combined[combined[field] != ""].sort_values(field).reset_index(drop=True)

    ids = combined["entity_id"].tolist()
    srcs = combined["src"].tolist()
    n = len(combined)

    for i in range(n):
        if srcs[i] != "s1":
            continue
        sid = ids[i]
        for j in range(i + 1, min(i + window + 1, n)):
            if srcs[j] == "oth":
                pairs.add((sid, ids[j]))
        for j in range(max(0, i - window), i):
            if srcs[j] == "oth":
                pairs.add((sid, ids[j]))

    logger.info("Sorted-neighbourhood (field=%s, W=%d): %d pairs",
                field, window, len(pairs))
    return pairs
