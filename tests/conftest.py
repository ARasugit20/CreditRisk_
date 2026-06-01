"""Shared pytest fixtures for CreditRisk_ pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import DATE_COLUMN, TARGET_COLUMN  # noqa: E402


@pytest.fixture
def minimal_lending_club_frame() -> pd.DataFrame:
    """Tiny synthetic LendingClub-like frame for fast unit tests (not real Kaggle data)."""

    rows: list[dict[str, object]] = []
    for index in range(30):
        charged_off = index % 2 == 1
        rows.append(
            {
                "loan_amnt": 8000.0 + index * 500,
                "annual_inc": 48000.0 + index * 1200,
                "dti": 10.0 + (index % 7),
                "fico_range_low": 650 + (index % 10) * 3,
                "fico_range_high": 654 + (index % 10) * 3,
                "revol_bal": 2000.0 + index * 150,
                "revol_util": 20.0 + (index % 5) * 8,
                "term": " 60 months" if index % 3 == 0 else " 36 months",
                "purpose": ["debt_consolidation", "credit_card", "home_improvement"][index % 3],
                "home_ownership": ["RENT", "MORTGAGE", "OWN"][index % 3],
                "emp_length": ["2 years", "5 years", "10+ years"][index % 3],
                DATE_COLUMN: f"Jan-{2014 + (index % 3)}",
                TARGET_COLUMN: "Charged Off" if charged_off else "Fully Paid",
                "funded_amnt": 8000.0 + index * 500,
                "int_rate": 8.0 + (index % 6),
                "grade": ["A", "B", "C", "D"][index % 4],
            }
        )
    return pd.DataFrame(rows)
