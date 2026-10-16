import numpy as np
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
    hold = evaluate_shadow_signal(candidate="x", signal="HOLD", timestamp="2026-10-16T12:00:00Z", risk_ok=True)
    invalid = evaluate_shadow_signal(candidate="x", signal="BUY NOW", timestamp="2026-10-16T12:00:00Z", risk_ok=True)
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


def test_timestamp_aligned_ab_uses_zero_for_no_trade_slots():
    from evaluation.shadow_ab import compare_ab_by_timestamp

    result = compare_ab_by_timestamp(
        ["2026-10-16T12:00:00Z", "2026-10-16T12:05:00Z", "2026-10-16T12:10:00Z"],
        [{"timestamp": "2026-10-16T12:00:00Z", "r": 1.0}],
        [{"timestamp": "2026-10-16T12:00:00Z", "r": 1.0}, {"timestamp": "2026-10-16T12:10:00Z", "r": 1.0}],
        min_samples=3,
    )
    assert result["sample_unit"] == "evaluation_timestamp"
    assert result["baseline"]["total_r"] == pytest.approx(1.0)
    assert result["candidate"]["total_r"] == pytest.approx(2.0)
    assert result["candidate"]["expectancy_r"] == pytest.approx(2 / 3)
    assert result["execution_enabled"] is False


def test_timestamp_alignment_normalizes_numpy_and_iso_utc_forms():
    from evaluation.shadow_ab import compare_ab_by_timestamp

    slot = np.datetime64("2026-10-16T12:00:00.000000000")
    result = compare_ab_by_timestamp(
        [slot],
        [{"timestamp": "2026-10-16T12:00:00Z", "r": 1.0}],
        [{"timestamp": "2026-10-16 13:00:00+01:00", "r": 2.0}],
        min_samples=1,
    )
    assert result["baseline"]["total_r"] == pytest.approx(1.0)
    assert result["candidate"]["total_r"] == pytest.approx(2.0)


def test_timestamp_aligned_ab_fails_closed_on_bad_alignment():
    from evaluation.shadow_ab import compare_ab_by_timestamp

    with pytest.raises(ValueError, match="unique"):
        compare_ab_by_timestamp(["2026-10-16T12:00:00Z", "2026-10-16 13:00:00+01:00"], [], [])
    with pytest.raises(ValueError, match="outside"):
        compare_ab_by_timestamp(["2026-10-16T12:00:00Z"], [{"timestamp": "2026-10-16T12:05:00Z", "r": 1.0}], [])
    with pytest.raises(ValueError, match="at most one"):
        compare_ab_by_timestamp(
            ["2026-10-16T12:00:00Z"],
            [{"timestamp": "2026-10-16T12:00:00Z", "r": 1.0}, {"timestamp": "2026-10-16 13:00:00+01:00", "r": -1.0}],
            [],
        )
