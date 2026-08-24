from __future__ import annotations

import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads_and_hybrid_weights_sum_to_one():
    config = load_config()
    assert config.hybrid_weights.total() == pytest.approx(1.0, abs=1e-6)


def test_bad_hybrid_weights_fail_fast(tmp_path):
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    raw["hybrid_weights"]["autoencoder"] = 5.0  # now sums to well over 1.0
    bad_config_path = tmp_path / "bad_config.yaml"
    with open(bad_config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh)

    with pytest.raises(ValueError, match="hybrid_weights must sum to 1.0"):
        load_config(bad_config_path)
