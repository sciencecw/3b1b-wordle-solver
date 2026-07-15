"""Gradio web app for the Reverse Wordle Solver.

Run locally:
    python wordle_app.py

Run with a public share link (72-hour tunnel):
    python wordle_app.py --share

For Hugging Face Spaces: rename or symlink this file to app.py, or set
CMD in the Dockerfile. The `share` flag is not needed on HF.
"""

from __future__ import annotations

import argparse

import gradio as gr

from src.pattern import get_pattern_matrix, pattern_to_int_list
from src.prior import get_true_wordle_prior, get_word_list
from src.reverse_solver import BETA_PRESETS, parse_share_text, reconstruct_guesses

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_BG = {0: "#787c7e", 1: "#c9b458", 2: "#6aaa64"}
TILE_STYLE = (
    "display:inline-flex;align-items:center;justify-content:center;"
    "width:44px;height:44px;border-radius:4px;margin:2px;"
    "font-size:22px;font-weight:bold;color:white;"
)

RATIONALITY_MAP = {"Casual": "casual", "Normal": "normal", "Expert": "expert"}

# Warm up on import so the first solve is fast
_PRIORS: dict | None = None
_ALLOWED: list | None = None


def _load():
    global _PRIORS, _ALLOWED
    if _PRIORS is None:
        _PRIORS = get_true_wordle_prior("wordle")
        _ALLOWED = set(get_word_list("wordle", short=False))
        # Warm the pattern matrix cache: the first get_pattern_matrix call loads
        # (or generates) the full grid, which is the slow part of a cold solve.
        first = next(iter(_ALLOWED))
        get_pattern_matrix([first], [first], "wordle")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _tile(v: int) -> str:
    emoji = ["⬛", "🟨", "🟩"][v]
    return f'<span style="{TILE_STYLE}background:{TILE_BG[v]}">{emoji}</span>'


def _row_html(pattern_int: int) -> str:
    return "".join(_tile(v) for v in pattern_to_int_list(pattern_int))


def _render_results(results: list[dict]) -> str:
    parts: list[str] = []
    for res in results[:3]:
        rank = res["rank"]
        pct = res["normalized_probability"] * 100

        medal = ["🥇", "🥈", "🥉"][rank - 1]
        header = (
            f'<div style="margin-top:24px;margin-bottom:8px;">'
            f'<span style="font-size:18px;font-weight:700;">'
            f'{medal} Reconstruction #{rank}</span>'
            f'<span style="margin-left:12px;font-size:14px;color:#666;">'
            f'relative confidence {pct:.0f}%</span>'
            f'</div>'
        )
        parts.append(header)

        for si in res["step_info"]:
            guess = si["guess"].upper()
            pat_html = _row_html(si["pattern"])
            prob = si["step_probability"]
            n_before = si["n_remaining_before"]
            n_after = si["n_remaining_after"]
            alts = si["top_alternatives"]

            alt_words = [w.upper() for w, _ in alts[1:4] if w.upper() != guess]
            alt_str = (
                f'<span style="color:#888;font-size:12px;font-style:italic;">'
                f'alt: {", ".join(alt_words)}</span>'
                if alt_words else ""
            )

            is_final = si["pattern"] == 242
            word_style = (
                "font-size:17px;font-weight:700;font-family:monospace;"
                "letter-spacing:1px;color:#333;"
            )
            prob_str = "" if is_final else f'p={prob:.0%}'
            remaining_str = (
                f'<span style="color:#888;font-size:12px;">'
                f'{n_before}→{n_after} words</span>'
            )

            row = (
                f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">'
                f'<span style="width:20px;color:#aaa;font-size:12px;">{si["step"]}.</span>'
                f'{pat_html}'
                f'<span style="{word_style}">{guess}</span>'
                f'<span style="font-size:12px;color:#555;">{prob_str}</span>'
                f'{remaining_str}'
                f'{alt_str}'
                f'</div>'
            )
            parts.append(row)

    container = (
        '<div style="font-family:sans-serif;max-width:680px;">'
        + "".join(parts)
        + "</div>"
    )
    return container


# ---------------------------------------------------------------------------
# Solver entrypoint
# ---------------------------------------------------------------------------


def solve(answer_raw: str, share_text: str, rationality_label: str, hard_mode: bool) -> str:
    _load()

    answer = answer_raw.strip().lower()
    if not answer:
        return '<p style="color:red;">Please enter the answer word.</p>'
    if answer not in _ALLOWED:
        return (
            f'<p style="color:red;">"{answer}" is not in the Wordle word list. '
            f"Check spelling.</p>"
        )

    try:
        patterns = parse_share_text(share_text)
    except ValueError as e:
        return f'<p style="color:red;">{e}</p>'

    beta = BETA_PRESETS[RATIONALITY_MAP.get(rationality_label, "normal")]

    try:
        results = reconstruct_guesses(
            patterns=patterns,
            answer=answer,
            game_name="wordle",
            beta=beta,
            priors=_PRIORS,
            hard_mode=hard_mode,
        )
    except ValueError as e:
        return f'<p style="color:red;">Solver error: {e}</p>'

    banner = ""
    if results and not results[0]["won"]:
        banner = (
            '<p style="color:#b45309;font-weight:600;">'
            f"❌ Lost game — none of the six guesses found {answer.upper()}. "
            "Reconstructing all six tries.</p>"
        )
    return banner + _render_results(results)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_EXAMPLE_4 = (
    "umbra",
    "Wordle 1,784 4/6\n\n🟨🟨⬛⬛⬛\n🟨⬛⬛🟨🟨\n⬛⬛🟨🟩🟨\n🟩🟩🟩🟩🟩",
    "Normal",
    False,
)
_EXAMPLE_5 = (
    "umbra",
    "Wordle 1,784 5/6\n\n⬜🟨🟨⬜⬜\n🟨🟨⬜⬜⬜\n🟨🟨🟨🟨⬜\n🟨🟨🟨🟨🟩\n🟩🟩🟩🟩🟩",
    "Normal",
    False,
)

with gr.Blocks(title="Reverse Wordle Solver") as app:
    gr.Markdown(
        """
# 🟩 Reverse Wordle Solver
Paste your Wordle share and enter the answer — the solver will reconstruct
the most likely guesses using information theory and a model of human play.

*Guesses favor the **2,315 official Wordle answer words**; obscure-but-valid
words are considered but discounted 10x. Uses a Boltzmann rationality model:
higher β = closer to optimal information-theoretic play.*
        """
    )

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer word",
            placeholder="e.g. umbra",
            scale=1,
        )
        share_box = gr.Textbox(
            label="Paste your Wordle share here",
            placeholder=(
                "Wordle 1,784 4/6\n\n"
                "🟨🟨⬛⬛⬛\n🟨⬛⬛🟨🟨\n⬛⬛🟨🟩🟨\n🟩🟩🟩🟩🟩"
            ),
            lines=8,
            scale=2,
        )

    with gr.Row():
        rationality = gr.Radio(
            ["Casual", "Normal", "Expert"],
            value="Normal",
            label="Player type",
            info="Casual (β=2): human-like  ·  Normal (β=5): engaged  ·  Expert (β=10): near-optimal",
        )
        hard_mode_box = gr.Checkbox(
            value=False,
            label="Hard mode",
            info="Every guess must reuse revealed hints (greens stay in place, yellows must appear)",
        )

    solve_btn = gr.Button("🔍 Reconstruct guesses", variant="primary", size="lg")
    output = gr.HTML(label="Reconstructions")

    gr.Examples(
        examples=[list(_EXAMPLE_4), list(_EXAMPLE_5)],
        inputs=[answer_box, share_box, rationality, hard_mode_box],
        label="Try these examples (Wordle 1,784, answer = UMBRA)",
    )

    solve_btn.click(
        fn=solve,
        inputs=[answer_box, share_box, rationality, hard_mode_box],
        outputs=output,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app.launch(
        share=args.share,
        server_port=args.port,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 720px !important; margin: auto; }",
    )
