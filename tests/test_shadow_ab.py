import pytest

from evaluation.shadow_ab import compare_ab, evaluate_shadow_signal


def test_shadow_signal_is_auditable_and_never_submits_order():
    record = evaluate_shadow_signal(
        candidate="candidate-v1",
        signal="BUY",
        timestamp="2026-10-16T12:00:00Z",
        risk_ok=True,
        spread=0.0001,
        max_spread=0.0002,
    )
    assert record["mode"] == "SHADOW_ONLY"
    assert record["eligible_for_shadow_review"] is True
    assert record["order_submitted"] is False
    assert record["live"] is False
    assert record["real"] is False


def test_shadow_signal_rejects_first_on_risk_or_spread_failure():
    record = evaluate_shadow_signal(
        candidate="candidate-v1",
        signal="SELL",
        timestamp="2026-10-16T12:00:00Z",
        risk_ok=False,
        spread=0.0004,
        max_spread=0.0002,
    )
    assert record["eligible_for_shadow_review"] is False
    assert "risk_check_failed" in record["reasons"]
    assert "spread_limit_exceeded" in record["reasons"]
    assert record["order_submitted"] is False


def test_hold_and_invalid_signals_are_not_eligible():
    hold = evaluate_shadow_signal(candidate="x", signal="HOLD", timestamp="t", risk_ok=True)
    invalid = evaluate_shadow_signal(candidate="x", signal="BUY NOW", timestamp="t", risk_ok=True)
    assert hold["eligible_for_shadow_review"] is False
    assert invalid["eligible_for_shadow_review"] is False
    assert "invalid_signal" in invalid["reasons"]


def test_ab_promotes_only_if_sample_and_candidate_checks_pass():
    result = compare_ab([0.01] * 120, [0.02] * 120)
    assert result["promotion_candidate"] is True
    assert result["paired_mean_delta_r"] == pytest.approx(0.01)
    assert result["execution_enabled"] is False
    assert result["live"] is False
    assert result["real"] is False


def test_ab_rejects_candidate_that_loses_or_has_too_few_samples():
    result = compare_ab([0.02] * 20, [-0.01] * 20, min_samples=100)
    assert result["promotion_candidate"] is False
    assert result["checks"]["minimum_sample"] is False
    assert result["checks"]["candidate_positive_expectancy"] is False
    assert result["checks"]["candidate_beats_baseline"] is False


def test_ab_requires_paired_finite_data():
    with pytest.raises(ValueError, match="equal length"):
        compare_ab([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        compare_ab([float("nan")], [1.0])
