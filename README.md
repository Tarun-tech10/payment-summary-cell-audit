# Payment summary cell audit

Working code for reading receipt photos and reconstructing their payment
summary: the number of purchased item rows, the printed totals and the
layout-grid cells that hold them.

A convolutional layout network predicts the grid, item count and fields; a
CRNN sequence reader (trained on real crops plus synthetic rendered text,
with CTC beam / n-best decoding) reads the amounts; gradient-boosted rankers
pick the final readings using arithmetic consistency checks.

Key files: `final2.py` (full pipeline), `prep.py`, `cnn*.py`, `gridcell.py`,
`crnn*.py`, `synth*.py`, `beamsearch.py`, `nbest.py`, `ranker.py`,
`build_submission.py`, `scorer.py`. `progress_*.txt` and `*.log` are run logs.

Final solution: https://github.com/Tarun-tech10/payment-summary-cell-audit-solution

The receipt images, model weights and cached arrays (`cache/`) and
submissions are not included.

Licensed under the MIT License.
