# tests/test_pipeline.py
import importlib.util
import sys
from pathlib import Path

import pytest

# Import the module under test via importlib (not a package, no __init__)
spec = importlib.util.spec_from_file_location(
    "pipeline",
    Path(__file__).parents[1] / "scripts/scaling/run_scaling_pipeline.py",
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_find_best_checkpoint_raises_when_missing(tmp_path, monkeypatch):
    """find_best_checkpoint raises FileNotFoundError when best.ckpt absent."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="best.ckpt not found"):
        pipeline.find_best_checkpoint("5m", "8m", is_sft=False)


def test_find_best_checkpoint_returns_absolute_path(tmp_path, monkeypatch):
    """find_best_checkpoint returns an absolute path when best.ckpt exists."""
    monkeypatch.chdir(tmp_path)
    ckpt_dir = tmp_path / "logs" / "tcr_5m_8m" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    best = ckpt_dir / "best.ckpt"
    best.write_text("fake")
    result = pipeline.find_best_checkpoint("5m", "8m", is_sft=False)
    assert result.is_absolute()
    assert result == best.resolve()


def test_find_best_sft_checkpoint_uses_sft_prefix(tmp_path, monkeypatch):
    """find_best_checkpoint with is_sft=True looks in logs/tcr_sft_{size}_{dataset}/."""
    monkeypatch.chdir(tmp_path)
    ckpt_dir = tmp_path / "logs" / "tcr_sft_1m_500k" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    best = ckpt_dir / "best.ckpt"
    best.write_text("fake")
    result = pipeline.find_best_checkpoint("1m", "500k", is_sft=True)
    assert result == best.resolve()


def test_parse_groups_file(tmp_path):
    """Groups file parsing handles comments and blank lines."""
    groups_file = tmp_path / "groups.txt"
    groups_file.write_text("0.1m,0\n# comment\n\n5m,2\n15m,3\n")
    groups = pipeline.parse_groups_file(groups_file)
    assert groups == [("0.1m", 0), ("5m", 2), ("15m", 3)]


def test_load_epitopes_from_file(tmp_path):
    """load_epitopes_from_file reads one epitope per non-blank line."""
    epi_file = tmp_path / "epitopes.txt"
    epi_file.write_text("GILGFVFTL\nNLVPMVATV\n\nELAGIGILTV\n")
    result = pipeline.load_epitopes_from_file(epi_file)
    assert result == ["GILGFVFTL", "NLVPMVATV", "ELAGIGILTV"]
