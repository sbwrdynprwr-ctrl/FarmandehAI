"""Deterministic research approval gates.

These gates are statistical promotion checks only. They never enable live trading.
"""

from __future__ import annotations


def validation_gate_details(report: dict) -> dict:
    """Return every hard-gate result so research decisions are auditable."""
    combined = report.get("combined", {})
    robustness = report.get("robustness", {})
    folds = robustness.get("folds", [])
    positive_folds = sum(
        1 for fold in folds
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

    checks = {
        "min_trades": combined.get("trades", 0) >= 100,
        "positive_total_r": combined.get("total_r", 0.0) > 0.0,
        "positive_expectancy": combined.get("expectancy", 0.0) > 0.0,
        "positive_folds": positive_folds >= 3,
        "neighbor_stability": (min(neighbor_rates) if neighbor_rates else 0.0) >= 0.50,
        "positive_5bps_sensitivity": spread_5bps > 0.0,
        "positive_monte_carlo_terminal": terminal_r > 0.0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "observed": {
            "trades": combined.get("trades", 0),
            "total_r": combined.get("total_r", 0.0),
            "expectancy": combined.get("expectancy", 0.0),
            "positive_folds": positive_folds,
            "min_neighbor_positive_rate": min(neighbor_rates) if neighbor_rates else 0.0,
            "spread_5bps_total_r": spread_5bps,
            "monte_carlo_terminal_r": terminal_r,
        },
    }


def passes_validation_gate(report: dict) -> bool:
    """Return True only when every hard research gate is satisfied."""
    return validation_gate_details(report)["pass"]
