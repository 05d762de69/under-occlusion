import matplotlib as mpl
from pathlib import Path

NATURE_RC = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
}

OKABE_ITO = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "sky":        "#56B4E9",
    "orange":     "#E69F00",
    "purple":     "#CC79A7",
}

MODEL_COLORS = {
    "shape_biased":   OKABE_ITO["blue"],
    "texture_biased": OKABE_ITO["vermillion"],
}

NATURE_SINGLE_COL = (2.8, 2.0)
NATURE_DOUBLE_COL = (5.7, 3.0)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def apply():
    mpl.rcParams.update(NATURE_RC)