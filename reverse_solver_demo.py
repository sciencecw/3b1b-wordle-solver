"""CLI demo for the reverse Wordle solver.

Given a Wordle result trace (pattern colors) and the correct answer, reconstructs
the most likely sequence of guesses the human player made.

Pattern formats accepted (space-separated on the command line):
  - Integer 0-242  (ternary encoding: MISS=0, MISPLACED=1, EXACT=2)
  - 5-digit list   e.g. 1,1,0,0,0
  - 5-emoji string e.g. 🟨🟨⬛⬛⬛

Example usage:
  # UMBRA example (4-guess game)
  python reverse_solver_demo.py --answer umbra --patterns 4 109 144 242

  # Emoji input
  python reverse_solver_demo.py --answer umbra \\
      --patterns "🟨🟨⬛⬛⬛" "🟨⬛⬛🟨🟨" "⬛⬛🟨🟩🟨" "🟩🟩🟩🟩🟩"

  # Casual player model, wider beam
  python reverse_solver_demo.py --answer crane --patterns 242 --rationality casual

  # Expert rationality, top-10 reconstructions
  python reverse_solver_demo.py --answer umbra --patterns 4 109 144 242 \\
      --rationality expert --top-k 10
"""

from __future__ import annotations

import argparse
import sys
import time

from src.prior import get_frequency_based_priors, get_true_wordle_prior
from src.reverse_solver import BETA_PRESETS, parse_pattern, reconstruct_guesses


def _parse_pattern_arg(s: str) -> int:
    """Parse a single pattern argument in any of the supported formats."""
    s = s.strip()
    # Emoji string
    if any(c in s for c in "⬛🟨🟩"):
        return parse_pattern(s)
    # Comma-separated list
    if "," in s:
        return parse_pattern([int(x) for x in s.split(",")])
    # Plain integer
    return parse_pattern(int(s))


def _print_result(result: dict, show_alternatives: bool = True) -> None:
    rank = result["rank"]
    guesses = result["guesses"]
    norm_p = result["normalized_probability"]
    log_p = result["log_probability"]

    print(f"  Rank #{rank}  —  relative probability {norm_p:.1%}  (log={log_p:.2f})")
    print(f"  Guesses: {' → '.join(guesses)}")
    for si in result["step_info"]:
        alts = si["top_alternatives"]
        remaining_info = (
            f"{si['n_remaining_before']} → {si['n_remaining_after']} possible"
        )
        p_str = f"{si['step_probability']:.3f}"
        line = (
            f"    Step {si['step']}: {si['guess'].upper():<8s} "
            f"{si['pattern_str']}  "
            f"p={p_str}  [{remaining_info}]"
        )
        if show_alternatives and len(alts) > 1:
            alt_str = "  alts: " + ", ".join(
                f"{w}({p:.2f})" for w, p in alts[1:4]
            )
            line += alt_str
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reverse Wordle solver: reconstruct guesses from a pattern trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--answer", required=True, help="The correct Wordle answer (e.g. umbra)"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        required=True,
        help=(
            "Pattern sequence. Each pattern can be: an int 0-242, a comma-separated "
            "5-digit list (e.g. 1,1,0,0,0), or a 5-emoji string. "
            "Last pattern must be all-green (242 / 2,2,2,2,2 / 🟩🟩🟩🟩🟩)."
        ),
    )
    parser.add_argument(
        "--game-name",
        choices=["wordle", "dungleon"],
        default="wordle",
        help="Game variant (default: wordle)",
    )
    rationality_group = parser.add_mutually_exclusive_group()
    rationality_group.add_argument(
        "--rationality",
        choices=list(BETA_PRESETS.keys()),
        default="normal",
        help=(
            "Human rationality preset: random (β=0), casual (β=2), "
            "normal (β=5, default), expert (β=10)"
        ),
    )
    rationality_group.add_argument(
        "--beta",
        type=float,
        help="Rationality parameter β directly (overrides --rationality)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=50,
        help="Beam search width (default: 50; larger = more thorough but slower)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of reconstructions to display (default: 5)",
    )
    parser.add_argument(
        "--first-guess-boost",
        type=float,
        default=5.0,
        help="Probability multiplier for known common openers at step 1 (default: 5.0)",
    )
    parser.add_argument(
        "--prior",
        choices=["wordle", "frequency"],
        default="wordle",
        help=(
            "Prior distribution: wordle (binary: in/out of official answer list, default) "
            "or frequency (sigmoid-weighted by word frequency, requires freq data file)"
        ),
    )
    parser.add_argument(
        "--hard-mode",
        action="store_true",
        help="Assume hard mode: each guess must be consistent with all prior hints",
    )
    parser.add_argument(
        "--no-alternatives",
        action="store_true",
        help="Suppress alternative guess suggestions in output",
    )
    args = parser.parse_args()

    beta = args.beta if args.beta is not None else BETA_PRESETS[args.rationality]
    rationality_label = (
        f"β={beta:.1f}" if args.beta is not None
        else f"{args.rationality} (β={beta})"
    )

    # Parse patterns
    try:
        patterns = [_parse_pattern_arg(p) for p in args.patterns]
    except (ValueError, KeyError) as e:
        print(f"Error parsing patterns: {e}", file=sys.stderr)
        sys.exit(1)

    # Load priors
    if args.prior == "frequency":
        priors = get_frequency_based_priors(args.game_name)
    else:
        priors = get_true_wordle_prior(args.game_name)

    # Print header
    from src.pattern import pattern_to_string
    pattern_display = "  ".join(pattern_to_string(p) for p in patterns)
    print()
    print(f"Reverse Wordle solver")
    print(f"  Answer:       {args.answer.upper()}")
    print(f"  Pattern trace ({len(patterns)} guesses):")
    for i, p in enumerate(patterns):
        print(f"    Guess {i + 1}: {pattern_to_string(p)}")
    print(f"  Rationality:  {rationality_label}")
    print(f"  Beam width:   {args.beam_width}")
    print(f"  Hard mode:    {'yes' if args.hard_mode else 'no'}")
    print()

    t0 = time.time()
    try:
        results = reconstruct_guesses(
            patterns=patterns,
            answer=args.answer.lower(),
            game_name=args.game_name,
            beta=beta,
            beam_width=args.beam_width,
            first_guess_boost=args.first_guess_boost,
            priors=priors,
            hard_mode=args.hard_mode,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - t0

    print(f"Top {min(args.top_k, len(results))} reconstructions  "
          f"(solved in {elapsed:.1f}s):\n")
    for result in results[: args.top_k]:
        _print_result(result, show_alternatives=not args.no_alternatives)


if __name__ == "__main__":
    main()
