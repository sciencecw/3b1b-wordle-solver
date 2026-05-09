"""Precompute forward-solver expected scores for the 2315 answer words as openers.

Run once: python precompute_opener_prior.py
Output:   data/wordle/opener_prior.json

Covers all 2315 answer words (the only words that appear as candidates by default).
Maps word -> log_prior where 0.0 = best opener (RAISE) and more negative = worse.
Non-answer words (rare fallback case) are not included; they receive ?? 0.0 in JS.
"""

import json
import numpy as np
from src.solver import get_expected_scores
from src.prior import get_word_list, get_true_wordle_prior

possible = get_word_list("wordle", short=True)   # 2315 answer words only
priors   = get_true_wordle_prior("wordle")

print(f"Computing expected scores for {len(possible)} answer words as openers...")
scores = get_expected_scores(possible, possible, priors, "wordle")

log_prior = -(scores - scores.min())   # 0.0 = best, more negative = worse

opener_prior = {w: round(float(lp), 3) for w, lp in zip(possible, log_prior)}

best5  = sorted(opener_prior.items(), key=lambda x: x[1], reverse=True)[:5]
worst5 = sorted(opener_prior.items(), key=lambda x: x[1])[:5]
print("Best openers :", best5)
print("Worst openers:", worst5)

out_path = "data/wordle/opener_prior.json"
with open(out_path, "w") as f:
    json.dump(opener_prior, f, separators=(",", ":"))
print(f"Saved {len(opener_prior)} entries to {out_path}")
