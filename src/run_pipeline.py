#!/usr/bin/env python3
"""
run_pipeline.py — Main entry point for the entity resolution pipeline.

Orchestrates the full workflow:
  1. Candidate generation (blocking)
  2. Model training on training data
  3. Test inference and output generation

Usage:
  python run_pipeline.py --data-dir ../../dataset --output-dir ../../output
"""

import argparse
import gc
import os
import sys
import time

# Ensure src/ is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to dataset/ directory")
    parser.add_argument("--output-dir", required=True,
                        help="Path to output/ directory")
    parser.add_argument("--sample-size", type=int, default=300000,
                        help="Training pairs to sample (default: 300k)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold")
    parser.add_argument("--model-path", default=None,
                        help="Path to pre-trained model bundle (e.g. models/merged_entity_model.joblib)")
    parser.add_argument("--skip-blocking", action="store_true",
                        help="Skip blocking (use existing candidate_pairs.tsv)")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_s1 = os.path.join(data_dir, "train", "train_source1.tsv")
    train_s2 = os.path.join(data_dir, "train", "train_source2.tsv")
    train_s3 = os.path.join(data_dir, "train", "train_source3.tsv")
    train_gt = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    test_s1 = os.path.join(data_dir, "test", "test_source1.tsv")
    test_s2 = os.path.join(data_dir, "test", "test_source2.tsv")
    test_s3 = os.path.join(data_dir, "test", "test_source3.tsv")

    candidate_pairs = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_results = os.path.join(output_dir, "matching_results.tsv")

    total_t0 = time.time()

    # Step 1: Candidate generation (blocking)
    if not args.skip_blocking:
        print("=" * 60)
        print("PHASE 1: Candidate Generation (Blocking)")
        print("=" * 60)
        from candidate_gen import run_candidate_generation
        run_candidate_generation(
            test_s1, test_s2, test_s3,
            candidate_pairs
        )
        gc.collect()
    else:
        print(f"Skipping blocking — using existing {candidate_pairs}")

    # Step 2: Model loading or training
    print("\n" + "=" * 60)
    print("PHASE 2: Matching Model")
    print("=" * 60)

    # Check for provided or default merged model bundle
    default_model = os.path.join(_HERE, "..", "models", "merged_entity_model.joblib")
    chosen_model_path = args.model_path or (default_model if os.path.exists(default_model) else None)

    if chosen_model_path and os.path.exists(chosen_model_path):
        import joblib
        from hybrid_matcher import MergedHybridEntityMatcher
        print(f"Loading merged hybrid model from: {chosen_model_path}")
        model = MergedHybridEntityMatcher.load(chosen_model_path)
        threshold = model.threshold
    else:
        from train import train_model
        model, threshold = train_model(
            train_s1, train_s2, train_s3, train_gt,
            sample_size=args.sample_size,
        )
        gc.collect()

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Threshold override: {threshold:.2f}")

    # Step 3: Test inference
    print("\n" + "=" * 60)
    print("PHASE 3: Test Inference")
    print("=" * 60)
    from inference import run_inference
    run_inference(
        model, threshold, candidate_pairs,
        test_s1, test_s2, test_s3,
        matching_results
    )

    elapsed = time.time() - total_t0
    print(f"\nPipeline complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
