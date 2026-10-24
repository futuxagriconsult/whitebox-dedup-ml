"""Tests for MARSTrainer and export_model_and_calibration."""
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from whitebox_dedup.trainer import MARSTrainer, export_model_and_calibration


def test_trainer_fit_and_calibrate():
    # Synthetic labelled pairs with jw_name_full
    data = {
        "jw_name_full": [0.1, 0.2, 0.3, 0.4, 0.88, 0.92, 0.95, 0.98, 0.99, 0.15],
        "is_duplicate": [0,   0,   0,   0,   1,    1,    1,    1,    1,    0],
    }
    df = pd.DataFrame(data)

    trainer = MARSTrainer(feature_cols=["jw_name_full"], model_name="test_trainer_v1")
    scorer, cal = trainer.fit(df, label_col="is_duplicate", alpha=0.05)

    assert scorer.model_name == "test_trainer_v1"
    assert len(scorer.hinges) > 0
    assert cal.alpha == 0.05
    assert 0.0 <= cal.threshold_duplicate <= 1.0
    assert 0.0 <= cal.threshold_distinct <= 1.0

    # High similarity pair should have high probability
    prob_high = scorer.score({"jw_name_full": 0.98})
    prob_low = scorer.score({"jw_name_full": 0.10})
    assert prob_high > prob_low

    with tempfile.TemporaryDirectory() as tmpdir:
        m_path, c_path = export_model_and_calibration(scorer, cal, tmpdir)
        assert Path(m_path).exists()
        assert Path(c_path).exists()
