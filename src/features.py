#!/usr/bin/env python3
"""
features.py — Similarity feature computation for candidate pairs.

Computes 10 features for each (S1, S2/S3) candidate pair:
  - Jaro-Winkler similarity on normalised names
  - Jaro-Winkler similarity on normalised addresses
  - Country agreement (binary)
  - Exact name agreement (binary)
  - Exact address agreement (binary)
  - Jaccard token overlap on names
  - Jaccard token overlap on addresses
  - Name length ratio (min/max)
  - Address length ratio (min/max)
  - Shared numeric address tokens (3+ digit tokens in common)
"""

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler


FEATURE_COLS = [
    "name_similarity", "address_similarity", "country_agreement",
    "exact_name_agreement", "exact_address_agreement", "token_overlap",
    "addr_token_overlap", "name_len_ratio", "addr_len_ratio",
    "shared_numeric_tokens",
]


def jw_similarity_array(texts_a, texts_b):
    """Compute Jaro-Winkler similarity for parallel arrays of strings."""
    n = len(texts_a)
    results = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a, b = texts_a[i], texts_b[i]
        if a and b:
            results[i] = JaroWinkler.similarity(a, b)
    return results


def jaccard_overlap_array(texts_a, texts_b):
    """Compute Jaccard token overlap for parallel arrays of strings."""
    n = len(texts_a)
    results = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a, b = texts_a[i], texts_b[i]
        t1 = set(a.split()) if a else set()
        t2 = set(b.split()) if b else set()
        union = t1 | t2
        if union:
            results[i] = len(t1 & t2) / len(union)
    return results


def shared_numeric_array(addr_a, addr_b):
    """
    Count shared numeric tokens (3+ digits) between address pairs.
    Captures matching ZIP/PIN codes and street numbers.
    """
    n = len(addr_a)
    results = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a, b = addr_a[i], addr_b[i]
        if a and b:
            nums_a = {t for t in a.split() if t.isdigit() and len(t) >= 3}
            nums_b = {t for t in b.split() if t.isdigit() and len(t) >= 3}
            if nums_a and nums_b:
                results[i] = len(nums_a & nums_b)
    return results


def compute_features(pairs_df, s1_lookup, other_lookup):
    """
    Compute all 10 features for a batch of candidate pairs.
    
    Parameters
    ----------
    pairs_df : DataFrame with columns [source1_entity_id, matched_entity_id]
    s1_lookup : DataFrame indexed by entity_id, columns [norm_name, norm_address, norm_country]
    other_lookup : DataFrame indexed by entity_id, columns [norm_name, norm_address, norm_country]
    
    Returns
    -------
    DataFrame with original ID columns plus all feature columns
    """
    # Join source 1 fields
    pairs_df = pairs_df.merge(
        s1_lookup[["norm_name", "norm_address", "norm_country"]],
        left_on="source1_entity_id", right_index=True, how="left"
    ).rename(columns={
        "norm_name": "name1", "norm_address": "addr1", "norm_country": "country1"
    })

    # Join source 2/3 fields
    pairs_df = pairs_df.merge(
        other_lookup[["norm_name", "norm_address", "norm_country"]],
        left_on="matched_entity_id", right_index=True, how="left"
    ).rename(columns={
        "norm_name": "name2", "norm_address": "addr2", "norm_country": "country2"
    })

    # Fill missing values
    for col in ["name1", "name2", "addr1", "addr2", "country1", "country2"]:
        pairs_df[col] = pairs_df[col].fillna("")

    n1 = pairs_df["name1"].values
    n2 = pairs_df["name2"].values
    a1 = pairs_df["addr1"].values
    a2 = pairs_df["addr2"].values

    # String similarity features
    pairs_df["name_similarity"] = jw_similarity_array(n1, n2)
    pairs_df["address_similarity"] = jw_similarity_array(a1, a2)

    # Exact match features
    pairs_df["country_agreement"] = (
        (pairs_df["country1"] == pairs_df["country2"]) &
        (pairs_df["country1"] != "")
    ).astype(np.float32)
    pairs_df["exact_name_agreement"] = (
        (pairs_df["name1"] == pairs_df["name2"]) &
        (pairs_df["name1"] != "")
    ).astype(np.float32)
    pairs_df["exact_address_agreement"] = (
        (pairs_df["addr1"] == pairs_df["addr2"]) &
        (pairs_df["addr1"] != "")
    ).astype(np.float32)

    # Token overlap features
    pairs_df["token_overlap"] = jaccard_overlap_array(n1, n2)
    pairs_df["addr_token_overlap"] = jaccard_overlap_array(a1, a2)

    # Length ratio features
    n1_len = np.array([len(x) for x in n1], dtype=np.float32)
    n2_len = np.array([len(x) for x in n2], dtype=np.float32)
    a1_len = np.array([len(x) for x in a1], dtype=np.float32)
    a2_len = np.array([len(x) for x in a2], dtype=np.float32)

    pairs_df["name_len_ratio"] = np.minimum(n1_len, n2_len) / np.maximum(
        np.maximum(n1_len, n2_len), 1.0
    )
    pairs_df["addr_len_ratio"] = np.minimum(a1_len, a2_len) / np.maximum(
        np.maximum(a1_len, a2_len), 1.0
    )

    # Shared numeric token count
    pairs_df["shared_numeric_tokens"] = shared_numeric_array(a1, a2)

    # Clean up intermediate columns
    pairs_df.drop(columns=["name1", "name2", "addr1", "addr2",
                           "country1", "country2"], inplace=True)

    return pairs_df
