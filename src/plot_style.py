"""Shared matplotlib style for new analysis modules."""

from __future__ import annotations

import matplotlib as mpl


def apply_plot_style() -> None:
    """Apply a consistent matplotlib style across analysis scripts."""

    mpl.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "font.family": "monospace",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
