from backtest.gates import passes_validation_gate


def valid_report():
    return {
        "combined": {"trades": 120, "total_r": 12.0, "expectancy": 0.10},
        "robustness": {
            "folds": [
                {"selected_oos": {"total_r": 2.0}, "neighbor_count": 10, "neighbor_positive_rate": 0.6},
                {"selected_oos": {"total_r": 3.0}, "neighbor_count": 10, "neighbor_positive_rate": 0.7},
                {"selected_oos": {"total_r": 4.0}, "neighbor_count": 10, "neighbor_positive_rate": 0.8},
                {"selected_oos": {"total_r": 3.0}, "neighbor_count": 10, "neighbor_positive_rate": 0.9},
            ],
            "spread_sensitivity": [
                {"spread": 0.00005, "metrics": {"total_r": 1.0}},
            ],
        },
        "monte_carlo": {"terminal_r": 12.0},
    }


def test_validation_gate_passes_only_when_all_hard_checks_pass():
    assert passes_validation_gate(valid_report())


def test_validation_gate_rejects_negative_five_bps_sensitivity():
    report = valid_report()
    report["robustness"]["spread_sensitivity"][0]["metrics"]["total_r"] = -0.01
    assert not passes_validation_gate(report)


def test_validation_gate_rejects_insufficient_positive_folds():
    report = valid_report()
    report["robustness"]["folds"][2]["selected_oos"]["total_r"] = -1.0
    report["robustness"]["folds"][3]["selected_oos"]["total_r"] = -1.0
    assert not passes_validation_gate(report)


def test_validation_gate_rejects_missing_neighbor_stability():
    report = valid_report()
    report["robustness"]["folds"][0]["neighbor_positive_rate"] = 0.49
    assert not passes_validation_gate(report)


def test_validation_gate_details_are_auditable():
    from backtest.gates import validation_gate_details
    details = validation_gate_details(valid_report())
    assert details["pass"] is True
    assert all(details["checks"].values())
    assert details["observed"]["positive_folds"] == 4
    assert details["observed"]["spread_5bps_total_r"] == 1.0
