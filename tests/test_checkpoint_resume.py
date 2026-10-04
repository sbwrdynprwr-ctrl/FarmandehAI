from backtest.research import _load_wf_checkpoint, _save_wf_checkpoint


def test_walk_forward_checkpoint_round_trip(tmp_path):
    path = tmp_path / "wf.json"
    payload = {
        "schema": 2,
        "hypothesis": "mean_reversion_v2",
        "df_len": 1000,
        "folds": 4,
        "train_ratio": 0.5,
        "selection_spread": 0.00005,
        "completed_folds": [
            {"fold": 1, "train_rows": 500, "oos_rows": 125, "params": {"hypothesis": "mean_reversion_v2"}, "selection": {"selection_score": [1, 2, 3, 4]}, "oos_metrics": {"trades": 10}}
        ],
    }
    _save_wf_checkpoint(str(path), payload)
    restored = _load_wf_checkpoint(str(path), "mean_reversion_v2", 1000, 4, 0.5)
    assert restored == payload


def test_walk_forward_checkpoint_rejects_stale_shape(tmp_path):
    path = tmp_path / "wf.json"
    payload = {
        "schema": 1,
        "hypothesis": "mean_reversion_v2",
        "df_len": 1000,
        "folds": 4,
        "train_ratio": 0.5,
        "completed_folds": [],
    }
    _save_wf_checkpoint(str(path), payload)
    assert _load_wf_checkpoint(str(path), "mean_reversion_v2", 1001, 4, 0.5) == {}
    assert _load_wf_checkpoint(str(path), "mean_reversion_rr", 1000, 4, 0.5) == {}
