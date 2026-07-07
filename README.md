# 3b1b's Wordle solver

[![Code Quality][codacy-image]][codacy]

This repository contains Python code to solve Wordle puzzles with information theory.

## Requirements

- Install the latest version of [Python 3.X][python-download-url] (at least version 3.10).
- Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To print an exhaustive list of command-line arguments, run:

```bash
python simulations.py --help
```

Choose the game with `--game-name`:

```bash
python simulations.py --game-name wordle
```

```bash
python simulations.py --game-name dungleon
```

Alternatively, run [`wordle_solver.ipynb`][colab-notebook]
[![Open In Colab][colab-badge]][colab-notebook]

## Reverse solver

The reverse solver does the opposite of the forward solver: given a finished
game's color trace (the emoji grid from a Wordle share) and the answer, it
reconstructs the most likely guesses the player made. It runs a beam search
over pattern-consistent candidate words, scored with a Boltzmann rationality
model (β controls how close to optimal the player is assumed to be), a
forward-solver word-quality prior, and a 10x discount for words outside the
official answer list. Answers outside the official list (e.g. newer NYT
answers) are supported. Hard mode (guesses must reuse revealed hints) is
supported via `--hard-mode` / the web checkbox.

```bash
# From a pattern trace (int, comma-list, or emoji forms accepted)
python reverse_solver_demo.py --answer umbra --patterns 4 109 144 242

# Emoji input, expert player model
python reverse_solver_demo.py --answer umbra \
    --patterns "🟨🟨⬛⬛⬛" "🟨⬛⬛🟨🟨" "⬛⬛🟨🟩🟨" "🟩🟩🟩🟩🟩" --rationality expert
```

Web interfaces:

- `python wordle_app.py` — Gradio app (paste a Wordle share, `requirements-app.txt`),
- `wordle_artifact.html` — self-contained HTML/JS page, no server needed.

The first ever solve generates `data/wordle/pattern_matrix.npy` (a few
minutes); subsequent solves take ~1 second. Run the tests with:

```bash
python -m pytest tests/
```

## Results

Results are shown [on the Wiki][wiki-results].

## References

- 3Blue1Brown, [*Solving Wordle using information theory*][youtube-video], posted on Youtube on February 6, 2022,
- [`3b1b/videos`][youtube-supplementary-code]: supplementary code (in Python) accompanying the aforementioned video,
- [`woctezuma/dungleon-bot`][dungleon-bot]: the application of different solvers to [Dungleon][dungleon-rules],
- [`woctezuma/Wordle-Bot`][wordle-bot-python-fork]: an extremely slow solver, mentioning some results.

<!-- Definitions -->

[codacy]: <https://www.codacy.com/gh/woctezuma/3b1b-wordle-solver/dashboard>
[codacy-image]: <https://app.codacy.com/project/badge/Grade/ff156cc6b4604ba1a7527448480a118a>

[python-download-url]: <https://www.python.org/downloads/>
[colab-notebook]: <https://colab.research.google.com/github/woctezuma/3b1b-wordle-solver/blob/colab/wordle_solver.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
[wiki-results]: <https://github.com/woctezuma/3b1b-wordle-solver/wiki>

[youtube-video]: <https://www.youtube.com/watch?v=v68zYyaEmEA>
[youtube-supplementary-code]: <https://github.com/3b1b/videos/tree/master/_2022/wordle>
[dungleon-bot]: <https://github.com/woctezuma/dungleon-bot>
[dungleon-rules]: <https://github.com/woctezuma/dungleon/wiki/Rules>
[wordle-bot-python-fork]: <https://github.com/woctezuma/Wordle-Bot>
