#!/usr/bin/env python3
"""
hybrid_matcher.py — Merged Hybrid Entity Matching Model.

Combines a Gradient Boosted Decision Tree trained on 4.8 million hard-negative
candidate pairs with domain precision guardrails:
1. Country Agreement Guard: Suppresses mismatched countries unless exact name matches.
2. Address Confidence Boost: Enhances confidence when exact address agreement and token overlap co-occur.
3. Exact Name Promotion: Promotes high-confidence links with exact name & country agreement.
"""

import os
import joblib
import numpy as np
import pandas as pd


class MergedHybridEntityMatcher:
    """
    Hybrid Entity Matcher combining hard-negative trained GBDT with precision guardrails.
    """

    def __init__(self, base_model, feature_cols=None, threshold=0.76, rules=None):
        self.base_model = base_model
        self.feature_cols = feature_cols or [
            "name_similarity", "address_similarity", "country_agreement",
            "exact_name_agreement", "exact_address_agreement", "token_overlap"
        ]
        self.threshold = threshold
        self.rules = rules or {
            "country_guard": 0.60,
            "addr_boost": 1.12,
            "name_boost": 1.08
        }

    def predict_proba(self, X_df):
        """Compute calibrated probability scores using base model and domain rules."""
        X_mat = X_df[self.feature_cols].values.astype(np.float32)
        base_probs = self.base_model.predict_proba(X_mat)[:, 1]

        country_agr = X_df["country_agreement"].values
        exact_name = X_df["exact_name_agreement"].values
        exact_addr = X_df["exact_address_agreement"].values
        tok_overlap = X_df["token_overlap"].values

        # Apply precision guardrails
        country_guard = np.where((country_agr == 0) & (exact_name == 0), self.rules["country_guard"], 1.0)
        addr_boost = np.where((exact_addr == 1) & (tok_overlap >= 0.5), self.rules["addr_boost"], 1.0)
        name_boost = np.where((exact_name == 1) & (country_agr == 1), self.rules["name_boost"], 1.0)

        calibrated = base_probs * country_guard * addr_boost * name_boost
        return np.clip(calibrated, 0.0, 1.0)

    def predict(self, X_df):
        """Return binary predictions using the precision-tuned threshold."""
        probs = self.predict_proba(X_df)
        return (probs >= self.threshold).astype(int)

    @classmethod
    def load(cls, model_path):
        """Load a saved model bundle."""
        bundle = joblib.load(model_path)
        return cls(
            base_model=bundle["model"],
            feature_cols=bundle.get("feature_cols"),
            threshold=bundle.get("optimal_threshold", 0.76),
            rules=bundle.get("rules")
        )

    def save(self, output_path):
        """Save the hybrid model bundle."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bundle = {
            "model": self.base_model,
            "feature_cols": self.feature_cols,
            "optimal_threshold": self.threshold,
            "rules": self.rules,
            "architecture": "Merged Hybrid: 4.8M Hard-Negative GBDT + Precision Guardrails"
        }
        joblib.dump(bundle, output_path)
