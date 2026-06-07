"""Shared, publication-ready figure styling for the position-curve plots.

The look is matched to the original *Lost in the Middle* paper (Liu et al.,
arXiv:2307.03172): a clean white-grid line plot, one line per setting, large
readable fonts, and a colour-blind-safe palette (Okabe–Ito) so the figures stay
legible in print and for readers with colour-vision deficiency.

Import `apply_paper_style()` once at the top of a plotting script, then use the
palette helpers so every model's figure shares the same colours and typography.
"""
import matplotlib.pyplot as plt

# Okabe–Ito colour-blind-safe palette. Stable mapping so a given document-count
# (or model) keeps the same colour across every figure in the paper.
DOC_COUNT_COLORS = {
    10: "#0072B2",   # blue
    20: "#E69F00",   # orange
    30: "#009E73",   # green
    50: "#CC79A7",   # reddish purple
    100: "#D55E00",  # vermillion
    200: "#56B4E9",  # sky blue
    500: "#F0E442",  # yellow
}

MODEL_COLORS = {
    "gpt-4.1": "#0072B2",
    "deepseek_deepseek-chat": "#E69F00",
    "gemini-2.5-flash": "#009E73",
    "claude-sonnet-4-6": "#CC79A7",
}
_FALLBACK_COLORS = ["#D55E00", "#56B4E9", "#999999", "#000000"]

# Reference-line styling, shared everywhere so readers learn it once.
ORACLE_STYLE = dict(color="#555555", linestyle="--", linewidth=1.6)
CLOSEDBOOK_STYLE = dict(color="#D55E00", linestyle=":", linewidth=1.8)


def apply_paper_style():
    """Apply the shared white-grid, large-font style used for every figure."""
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style in plt.style.available:
            plt.style.use(style)
            break
    plt.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "lines.markeredgewidth": 0.8,
    })


def fallback_color_iter():
    """Iterator over fallback colours for models/settings without a fixed colour."""
    return iter(_FALLBACK_COLORS)
