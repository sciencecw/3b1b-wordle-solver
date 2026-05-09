"""Precompute forward-solver expected scores for all allowed first guesses.

Run once: python precompute_opener_prior.py
Output:   data/wordle/opener_prior.json

Maps word -> log_prior in natural units (expected guesses below optimal):
  0.0   = best opener (RAISE/SOARE tier)
  -0.10 = threshold; words below this are excluded from the dict
  Words not in the dict get a default of -0.2 in the JS artifact
  (approximate median of excluded words' actual values).
"""

import json
import numpy as np
from src.solver import get_expected_scores
from src.prior import get_word_list, get_true_wordle_prior

THRESHOLD = -0.10

allowed  = get_word_list("wordle", short=False)
possible = get_word_list("wordle", short=True)
priors   = get_true_wordle_prior("wordle")

print(f"Computing expected scores for {len(allowed)} words...")
scores = get_expected_scores(allowed, possible, priors, "wordle")

log_prior = -(scores - scores.min())   # 0.0 = best, more negative = worse

opener_prior = {
    w: round(float(lp), 3)
    for w, lp in zip(allowed, log_prior)
    if lp >= THRESHOLD
}

print(f"Kept {len(opener_prior)} / {len(allowed)} words (threshold={THRESHOLD})")
best5 = sorted(opener_prior.items(), key=lambda x: x[1], reverse=True)[:5]
worst5 = sorted(opener_prior.items(), key=lambda x: x[1])[:5]
print("Best openers :", best5)
print("Worst in dict:", worst5)

excluded = [lp for lp in log_prior if lp < THRESHOLD]
print(f"Excluded median log_prior: {np.median(excluded):.3f}  (use as JS default)")

out_path = "data/wordle/opener_prior.json"
with open(out_path, "w") as f:
    json.dump(opener_prior, f, separators=(",", ":"))
print(f"Saved {len(opener_prior)} entries to {out_path}")
