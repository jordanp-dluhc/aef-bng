"""Tests for aef_bng.cli."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from aef_bng.cli import main


@pytest.mark.unit
class TestCli:
    """Tests for the Click CLI entry point."""

    def test_main_help(self) -> None:
        """--help prints usage and exits 0."""
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AEF embeddings" in result.output

    def test_process_help(self) -> None:
        """process --help lists all options."""
        result = CliRunner().invoke(main, ["process", "--help"])
        assert result.exit_code == 0
        assert "--year" in result.output
        assert "--output" in result.output
        assert "--bounds" in result.output

    def test_process_requires_year(self) -> None:
        """process without --year exits with a non-zero code."""
        result = CliRunner().invoke(main, ["process", "--output", "/tmp/out"])  # noqa: S108
        assert result.exit_code != 0

    def test_process_runs_pipeline_and_prints_rows(self, tmp_path) -> None:
        """process invokes run_pipeline and prints per-year row counts."""
        with patch("aef_bng.pipeline.run_pipeline", new_callable=AsyncMock) as mock:
            mock.return_value = {2024: 1_000_000}
            result = CliRunner().invoke(
                main,
                [
                    "process",
                    "--year",
                    "2024",
                    "--bounds",
                    "530000",
                    "180000",
                    "540000",
                    "190000",
                    "--output",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0
        assert "1,000,000" in result.output

    def test_verbose_flag_accepted(self, tmp_path) -> None:
        """--verbose flag is accepted without error."""
        with patch("aef_bng.pipeline.run_pipeline", new_callable=AsyncMock) as mock:
            mock.return_value = {2024: 0}
            result = CliRunner().invoke(
                main,
                ["--verbose", "process", "--year", "2024", "--output", str(tmp_path)],
            )
        assert result.exit_code == 0
