#!/usr/bin/env python3
"""
candidate_gen.py — Multi-pass blocking pipeline for candidate generation.

Runs all enabled blocking strategies on test source files and writes
candidate_pairs.tsv. This module mirrors the blocking strategies from
the candidate_generation/ package but is self-contained.
"""

import gc
import os
import time
from collections import defaultdict

import pandas as pd

from preprocessing import load_source_chunked, preprocess_dataframe
from blocking import (
    block_by_exact_name,
    block_by_address_country,
    block_by_rare_token,
    block_by_postal_prefix,
    block_by_sorted_neighbourhood,
)


# Common tokens that should not be used as blocking keys
COMMON_TOKENS_DENYLIST = {
    "limited", "private", "llc", "inc", "ltd", "pvt", "center", "com",
    "partners", "services", "group", "corp", "and", "india", "care",
    "holdings", "associates", "llp", "corporation", "health", "the",
    "of", "co", "company", "enterprises", "international", "national",
    "global", "solution", "solutions", "management", "trading", "general",
    "new", "business", "consulting", "technology", "technologies", "retail",
    "restaurant", "foods", "food", "store", "shop", "market",
    "construction", "realty", "estate", "real", "auto", "automotive",
}

ADDRESS_TOKEN_DENYLIST = COMMON_TOKENS_DENYLIST | {
    "street", "road", "avenue", "drive", "boulevard", "lane", "court",
    "place", "parkway", "highway", "freeway", "square", "suite", "floor",
    "north", "south", "east", "west", "ne", "nw", "se", "sw",
    "plot", "sector", "block", "phase", "colony", "nagar", "near",
    "main", "cross", "layout", "extension", "village", "town",
    "district", "state", "city", "area", "zone", "ward",
}


MAX_BLOCK_SIZE = 30
RARE_TOKEN_MAX_DOCS = 50
RARE_TOKEN_MAX_DOCS_ADDR = 15
RARE_TOKEN_MAX_BLOCK_ADDR = 20
SNM_WINDOW = 5


def run_candidate_generation(s1_path, s2_path, s3_path, output_path):
    """Execute the full blocking pipeline and write candidate_pairs.tsv."""
    t0 = time.time()

    print("Loading sources ...")
    s1 = load_source_chunked(s1_path)
    s2 = load_source_chunked(s2_path)
    s3 = load_source_chunked(s3_path)

    source1_ids = set(s1["entity_id"])
    strategy_pairs = {}

    # A: Exact normalised name
    print("[A] Exact name blocking ...")
    p = block_by_exact_name(s1, s2, MAX_BLOCK_SIZE) | \
        block_by_exact_name(s1, s3, MAX_BLOCK_SIZE)
    strategy_pairs["name_exact"] = p
    print(f"    {len(p):,} pairs")

    # B: Exact address + country
    print("[B] Address+country blocking ...")
    p = block_by_address_country(s1, s2, MAX_BLOCK_SIZE) | \
        block_by_address_country(s1, s3, MAX_BLOCK_SIZE)
    strategy_pairs["address_country"] = p
    print(f"    {len(p):,} pairs")

    # C: Rare-token name
    print("[C] Rare-token name blocking ...")
    p = block_by_rare_token(s1, s2, "norm_name", COMMON_TOKENS_DENYLIST,
                            RARE_TOKEN_MAX_DOCS, MAX_BLOCK_SIZE) | \
        block_by_rare_token(s1, s3, "norm_name", COMMON_TOKENS_DENYLIST,
                            RARE_TOKEN_MAX_DOCS, MAX_BLOCK_SIZE)
    strategy_pairs["rare_token_name"] = p
    print(f"    {len(p):,} pairs")

    # D: Rare-token address (tight thresholds)
    print("[D] Rare-token address blocking ...")
    p = block_by_rare_token(s1, s2, "norm_address", ADDRESS_TOKEN_DENYLIST,
                            RARE_TOKEN_MAX_DOCS_ADDR, RARE_TOKEN_MAX_BLOCK_ADDR) | \
        block_by_rare_token(s1, s3, "norm_address", ADDRESS_TOKEN_DENYLIST,
                            RARE_TOKEN_MAX_DOCS_ADDR, RARE_TOKEN_MAX_BLOCK_ADDR)
    strategy_pairs["rare_token_address"] = p
    print(f"    {len(p):,} pairs")

    # E: Postal prefix
    print("[E] Postal prefix blocking ...")
    p = block_by_postal_prefix(s1, s2, 100) | \
        block_by_postal_prefix(s1, s3, 100)
    strategy_pairs["postal_prefix"] = p
    print(f"    {len(p):,} pairs")

    # G: Sorted neighbourhood
    print("[G] Sorted neighbourhood blocking ...")
    p = block_by_sorted_neighbourhood(s1, s2, "norm_name", SNM_WINDOW) | \
        block_by_sorted_neighbourhood(s1, s3, "norm_name", SNM_WINDOW)
    strategy_pairs["sorted_neighbourhood"] = p
    print(f"    {len(p):,} pairs")

    # Combine all strategies
    all_pairs = set()
    for name, pairs in strategy_pairs.items():
        before = len(all_pairs)
        all_pairs |= pairs
        added = len(all_pairs) - before
        print(f"  {name:<25} {len(pairs):>10,} → cumulative {len(all_pairs):>10,} (+{added})")

    print(f"\nTotal unique candidate pairs: {len(all_pairs):,}")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapping = defaultdict(set)
    for s1_id, cid in all_pairs:
        if cid.startswith(("S2-", "S3-")):
            mapping[s1_id].add(cid)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(source1_ids):
            cands = mapping.get(s1_id)
            if cands:
                f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")
            else:
                f.write(f"{s1_id}\t\n")

    elapsed = time.time() - t0
    print(f"Candidate generation done in {elapsed:.1f}s")
    print(f"Written: {output_path}")
