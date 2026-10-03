# Business Entity Resolution — AWS ML Challenge 2026

A machine-learning pipeline that resolves business entity records across three independent, noisy data sources. Given records with no common identifiers, it determines which records refer to the same real-world business.

## Problem Statement

Business identity data arrives from multiple independent sources — each contributing partial, noisy fragments. Records share no common identifiers. The goal is to find all matching records from **Source 2** and **Source 3** for each **Source 1** entity.

**Evaluation metric:** Macro-averaged F₀.₅ (precision-weighted — false merges penalised 2× over missed matches).

---

## Architecture

The pipeline runs in three sequential phases:

### Phase 1 — Candidate Generation (Blocking)

Reduces the O(n×m) comparison space using six complementary strategies, achieving **72.48% recall** over 54M candidate pairs:

| ID | Strategy | Key Idea |
|----|----------|----------|
| A | **Exact name** | Identical normalised business names |
| B | **Address + country** | Same normalised address and country |
| C | **Rare-token name** | Shared uncommon name tokens (≤ 50 docs) |
| D | **Rare-token address** | Shared uncommon address tokens (≤ 15 docs) |
| E | **Postal prefix** | Shared ZIP/PIN code prefix + country |
| G | **Sorted neighbourhood** | Adjacent records sorted by name (window = 5) |

All strategies guard against quadratic blowup with per-block size limits.

### Phase 2 — Merged Hybrid Matching Model

- **Base learner:** Regularised Gradient Boosted Decision Tree (XGBoost) trained on **4.8 million hard-negative** candidate pairs.
- **10 similarity features:** Jaro-Winkler on names/addresses, Jaccard token overlap, country agreement, exact match flags, length ratios, shared numeric tokens.
- **Precision guardrails** applied post-prediction:
  1. *Country Agreement Guard* — suppresses cross-country links unless names match exactly.
  2. *Address Confidence Boost* — promotes pairs with exact address + token overlap ≥ 0.5.
  3. *Exact Name Promotion* — calibrates confidence on identical normalised names.
- **Threshold calibration:** Optimised directly for macro F₀.₅ on a held-out validation split.

### Phase 3 — Inference

Scores all candidate pairs in memory-safe batches (500k pairs/batch). Writes `matching_results.tsv` with every Source 1 entity (including singletons).

---

## Repository Structure

```
business-entity-resolution/
├── src/
│   ├── run_pipeline.py      # Main entry point — orchestrates all three phases
│   ├── preprocessing.py     # Name / address normalisation (diacritics, legal suffixes, abbreviations)
│   ├── blocking.py          # Six blocking strategies
│   ├── candidate_gen.py     # Blocking orchestration — runs all strategies, writes candidate_pairs.tsv
│   ├── features.py          # 10 similarity features for candidate pairs
│   ├── train.py             # XGBoost training + F₀.₅-optimised threshold selection
│   ├── inference.py         # Batched test scoring + output writing
│   └── hybrid_matcher.py   # MergedHybridEntityMatcher with precision guardrails
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note on models:** The pre-trained model bundle (`merged_entity_model.joblib`, ~2.6 MB) is excluded from this repository. Either train from scratch using the instructions below or download it separately.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/business-entity-resolution.git
cd business-entity-resolution

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

---

## Data Layout

Place the competition dataset in a `dataset/` directory at the same level as this repo (or pass the path via `--data-dir`):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files are **tab-separated** (`.tsv`). Read with `pd.read_csv(..., sep="\t")`.

---

## Running the Pipeline

### Option A — Full pipeline (blocking + training + inference)

```bash
cd src/
python run_pipeline.py \
    --data-dir ../../dataset \
    --output-dir ../../output
```

### Option B — Skip blocking (use existing `candidate_pairs.tsv`)

If `candidate_pairs.tsv` already exists in your output directory:

```bash
cd src/
python run_pipeline.py \
    --data-dir ../../dataset \
    --output-dir ../../output \
    --skip-blocking
```

### Option C — Use a pre-trained model bundle

```bash
cd src/
python run_pipeline.py \
    --data-dir ../../dataset \
    --output-dir ../../output \
    --model-path ../models/merged_entity_model.joblib \
    --skip-blocking
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | *(required)* | Path to directory containing `train/` and `test/` |
| `--output-dir` | *(required)* | Where to write `candidate_pairs.tsv` and `matching_results.tsv` |
| `--sample-size` | 300,000 | Training pairs to sample |
| `--threshold` | auto | Override the decision threshold |
| `--model-path` | auto | Path to pre-trained `.joblib` bundle |
| `--skip-blocking` | false | Reuse an existing `candidate_pairs.tsv` |

---

## Output Files

| File | Description |
|------|-------------|
| `output/matching_results.tsv` | Final entity matches — upload to leaderboard |
| `output/candidate_pairs.tsv` | Blocking candidate set (for pipeline verification) |

### Validate before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Prints `PASS` (exit 0) or a numbered list of issues to fix (exit 1).

---

## Expected Runtime (16 GB RAM machine)

| Phase | Time |
|-------|------|
| Candidate generation | ~30 min |
| Model training | ~15 min |
| Test inference | ~60–90 min |
| **Total** | **~2 hours** |

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `name_similarity` | Jaro-Winkler similarity on normalised business names |
| `address_similarity` | Jaro-Winkler similarity on normalised addresses |
| `country_agreement` | Binary — same country |
| `exact_name_agreement` | Binary — identical normalised names |
| `exact_address_agreement` | Binary — identical normalised addresses |
| `token_overlap` | Jaccard token overlap on names |
| `addr_token_overlap` | Jaccard token overlap on addresses |
| `name_len_ratio` | min/max character length ratio on names |
| `addr_len_ratio` | min/max character length ratio on addresses |
| `shared_numeric_tokens` | Count of shared 3+ digit numeric tokens (ZIP, PIN, street numbers) |

---

## Preprocessing

The normalisation pipeline (`preprocessing.py`) handles:

- **Diacritic stripping** (`café → cafe`)
- **Legal suffix canonicalisation** (`Private Limited → pvt ltd`, `Corporation → corp`)
- **Junk prefix removal** (`***, --, M/s`)
- **Address abbreviation expansion** (`St → street`, `Ave → avenue`)
- **Whitespace / punctuation normalisation**

---

## License

MIT
# AWS-Ml-Challenge
