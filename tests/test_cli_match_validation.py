"""Tests for cli_match's same-name-refusal check (__main__.py).

The matching algorithms use the source/target mask names as the dimension
names of the scores, so identical names crash deep inside matching (the
dict literal coords={source_name: ..., target_name: ...} collapses to one
key). Identical source/target is invalid input anyway (matching a section
against itself), so cli_match refuses it outright, before reading any file.
See also tests/matching/test_identical_source_target_name.py, which covers
the dimension names at the algorithm level.
"""

from __future__ import annotations

import numpy as np
import tifffile
from click.testing import CliRunner

from spellmatch.__main__ import cli


def _write_mask(path, labels):
    tifffile.imwrite(path, np.array(labels, dtype=np.uint16))


def _simple_mask():
    # A tiny 2-cell labeled mask: label 1 in the top-left, label 2 in the
    # bottom-right, background 0 elsewhere.
    arr = np.zeros((10, 10), dtype=np.uint16)
    arr[1:3, 1:3] = 1
    arr[6:8, 6:8] = 2
    return arr


def test_refuses_identical_mask_filenames(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    source_mask = source_dir / "same.tiff"
    target_mask = target_dir / "same.tiff"
    _write_mask(source_mask, _simple_mask())
    _write_mask(target_mask, _simple_mask())

    result = CliRunner().invoke(
        cli,
        [
            "match",
            str(source_mask),
            str(target_mask),
            "--algorithm",
            "icp",
            str(tmp_path / "scores.nc"),
        ],
    )

    assert result.exit_code != 0
    assert "same filename" in result.output
    assert "same.tiff" in result.output


def test_refuses_identical_image_filenames(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    source_mask = source_dir / "source_mask.tiff"
    target_mask = target_dir / "target_mask.tiff"
    _write_mask(source_mask, _simple_mask())
    _write_mask(target_mask, _simple_mask())
    source_img = source_dir / "same_img.tiff"
    target_img = target_dir / "same_img.tiff"
    tifffile.imwrite(source_img, _simple_mask().astype(np.float32))
    tifffile.imwrite(target_img, _simple_mask().astype(np.float32))

    result = CliRunner().invoke(
        cli,
        [
            "match",
            str(source_mask),
            str(target_mask),
            "--source-image",
            str(source_img),
            "--target-image",
            str(target_img),
            "--algorithm",
            "icp",
            str(tmp_path / "scores.nc"),
        ],
    )

    assert result.exit_code != 0
    assert "same filename" in result.output
    assert "same_img.tiff" in result.output


def test_distinctly_named_masks_are_matched(tmp_path):
    source_mask = tmp_path / "source_mask.tiff"
    target_mask = tmp_path / "target_mask.tiff"
    _write_mask(source_mask, _simple_mask())
    _write_mask(target_mask, _simple_mask())

    result = CliRunner().invoke(
        cli,
        [
            "match",
            str(source_mask),
            str(target_mask),
            "--algorithm",
            "icp",
            str(tmp_path / "scores.nc"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "scores.nc").exists()
