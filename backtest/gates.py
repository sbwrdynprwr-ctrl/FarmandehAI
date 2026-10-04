"""Deterministic research approval gates.

These gates are statistical promotion checks only. They never enable live trading.
"""

from __future__ import annotations


def passes_validation_gate(report: dict) -> bool:
    """Return True only when every hard research gate is satisfied."""
    combined = report.get("combined", {})
    robustness = report.get("robustness", {})
    folds = robustness.get("folds", [])
    positive_folds = sum(
        1
        for fold in folds
        if fold.get("selected_oos", {}).get("total_r", 0.0) > 0.0
    )
    neighbor_rates = [
        fold.get("neighbor_positive_rate", 0.0)
        for fold in folds
        if fold.get("neighbor_count", 0)
    ]
    spread_5bps = sum(
        item.get("metrics", {}).get("total_r", 0.0)
        for item in robustness.get("spread_sensitivity", [])
        if item.get("spread") == 0.00005
    )
    terminal_r = report.get("monte_carlo", {}).get("terminal_r", 0.0)

    return (
        combined.get("trades", 0) >= 100
        and combined.get("total_r", 0.0) > 0.0
        and combined.get("expectancy", 0.0) > 0.0
        and positive_folds >= 3
        and (min(neighbor_rates) if neighbor_rates else 0.0) >= 0.50
        and spread_5bps > 0.0
        and terminal_r > 0.0
    )
