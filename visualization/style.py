import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, Union

# IEEE Publication Standards
# Column width: 3.5 inches
# Double column width: 7.16 inches

IEEE_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.5, 2.625),  # 4:3 aspect ratio for single column
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLOR_PALETTE = {
    "primary": "#1f77b4",     # Blue
    "secondary": "#ff7f0e",   # Orange
    "tertiary": "#2ca02c",    # Green
    "quaternary": "#d62728",  # Red
    "quinary": "#9467bd",     # Purple
    "senary": "#8c564b",      # Brown
    "f0_color": "#1f77b4",    # Blue
    "f1_color": "#ff7f0e",    # Orange
    "f2_color": "#2ca02c",    # Green
    "stable": "#2ca02c",      # Green
    "unstable": "#d62728",    # Red
    "abstraction": "#9467bd", # Purple
    "memory": "#8c564b"       # Brown
}

LINEWIDTH = 1.0
MARKERSIZE = 4.0

def set_ieee_style() -> None:
    """Applies the IEEE publication style to matplotlib."""
    for key, value in IEEE_RCPARAMS.items():
        mpl.rcParams[key] = value

def get_color(name: str) -> str:
    """
    Returns the hex color code for a given named color in the palette.
    
    Args:
        name: Name of the color.
        
    Returns:
        Hex color string.
    """
    if name in COLOR_PALETTE:
        return COLOR_PALETTE[name]
    # Fallback
    return COLOR_PALETTE.get("primary", "#000000")
