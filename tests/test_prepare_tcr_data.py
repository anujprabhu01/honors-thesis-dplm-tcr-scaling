# tests/test_prepare_tcr_data.py
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from datasets import load_from_disk


def _make_pickle(path: Path, n: int = 200):
    """Write a minimal TCR pickle file."""
    seqs = [f"CASS{'A' * (10 + i % 5)}QYF" for i in range(n)]
    df = pd.DataFrame({"tcr": seqs})
    with open(path, "wb") as f:
        pickle.dump(df, f)


def test_cli_produces_dataset(tmp_path):
    """Running prepare_tcr_data.py with CLI args creates a valid HF dataset."""
    pkl = tmp_path / "seqs.pkl"
    _make_pickle(pkl, n=100)
    out = tmp_path / "out_dataset"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/scaling/prepare_tcr_data.py",
            "--source_path", str(pkl),
            "--num_samples", "80",
            "--output_dir", str(out),
            "--seed", "42",
        ],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    ds = load_from_disk(str(out))
    assert "train" in ds
    assert "valid" in ds
    assert len(ds["train"]) == 64  # 80 * 0.80
    assert len(ds["valid"]) == 16  # 80 * 0.20


def test_cli_default_seed_is_reproducible(tmp_path):
    """Same seed produces same dataset ordering."""
    pkl = tmp_path / "seqs.pkl"
    _make_pickle(pkl, n=100)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in [out1, out2]:
        result = subprocess.run(
            [sys.executable, "scripts/scaling/prepare_tcr_data.py",
             "--source_path", str(pkl), "--num_samples", "80",
             "--output_dir", str(out), "--seed", "42"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    ds1 = load_from_disk(str(out1))
    ds2 = load_from_disk(str(out2))
    assert ds1["train"]["seq"] == ds2["train"]["seq"]
