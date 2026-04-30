"""Tests for aef_bng.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

from aef_bng.config import AEFBNGConfig
from aef_bng.constants import AEF_NODATA, AEF_NUM_BANDS
from aef_bng.grid import ChunkSpec
from aef_bng.pipeline import process_chunk, process_year, run_pipeline


@pytest.mark.unit
class TestProcessChunk:
    """Tests for single chunk processing."""

    @pytest.mark.asyncio
    async def test_no_tiles_returns_none(self) -> None:
        """Chunk with no matching tiles returns None."""
        config = AEFBNGConfig(years=[2024])
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        index = MagicMock()
        index.tiles_for_chunk.return_value = []

        result = await process_chunk(chunk, 2024, index, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_all_nodata_returns_none(self) -> None:
        """Chunk where all reprojected data is nodata returns None."""
        config = AEFBNGConfig(years=[2024])
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        index = MagicMock()
        index.tiles_for_chunk.return_value = [
            {"path": "test.tif", "crs": "EPSG:32630", "utm_bounds": None, "year": 2024}
        ]

        nodata_array = np.full((AEF_NUM_BANDS, 1000, 1000), AEF_NODATA, dtype=np.int8)

        with (
            patch("aef_bng.pipeline.read_tile", new_callable=AsyncMock) as mock_read,
            patch("aef_bng.pipeline.reproject_tile_to_bng") as mock_reproject,
        ):
            from affine import Affine

            mock_read.return_value = (nodata_array, Affine.identity(), "EPSG:32630")
            mock_reproject.return_value = nodata_array

            result = await process_chunk(chunk, 2024, index, config)
            assert result is None

    @pytest.mark.asyncio
    async def test_valid_data_returns_table(self) -> None:
        """Chunk with valid data returns an Arrow table."""
        config = AEFBNGConfig(years=[2024])
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        index = MagicMock()
        index.tiles_for_chunk.return_value = [
            {"path": "test.tif", "crs": "EPSG:32630", "utm_bounds": None, "year": 2024}
        ]

        valid_array = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)

        with (
            patch("aef_bng.pipeline.read_tile", new_callable=AsyncMock) as mock_read,
            patch("aef_bng.pipeline.reproject_tile_to_bng") as mock_reproject,
        ):
            from affine import Affine

            mock_read.return_value = (valid_array, Affine.identity(), "EPSG:32630")
            mock_reproject.return_value = valid_array

            result = await process_chunk(chunk, 2024, index, config)
            assert result is not None
            assert isinstance(result, pa.Table)
            assert result.num_rows == 1_000_000

    @pytest.mark.asyncio
    async def test_tile_exception_is_caught(self) -> None:
        """Exception during tile read is caught; chunk returns None if all tiles fail."""
        config = AEFBNGConfig(years=[2024])
        chunk = ChunkSpec(
            bng_10km_ref="TQ38",
            bounds_bng=(530_000, 180_000, 540_000, 190_000),
            bounds_wgs84=(-0.15, 51.48, -0.01, 51.57),
        )
        index = MagicMock()
        index.tiles_for_chunk.return_value = [
            {"path": "bad.tif", "crs": "EPSG:32630", "year": 2024}
        ]

        with patch("aef_bng.pipeline.read_tile", new_callable=AsyncMock) as mock_read:
            mock_read.side_effect = RuntimeError("S3 connection error")
            result = await process_chunk(chunk, 2024, index, config)
        assert result is None  # all tiles failed → no reprojected data


@pytest.mark.unit
class TestProcessYear:
    """Tests for process_year orchestration."""

    @pytest.mark.asyncio
    async def test_no_data_returns_zero(self, tmp_path) -> None:
        """process_year returns 0 when all chunks produce no data."""
        config = AEFBNGConfig(
            years=[2024],
            bounds=(530_000, 180_000, 540_000, 190_000),
            output_path=str(tmp_path),
        )
        index = MagicMock()
        index.tiles_for_chunk.return_value = []
        result = await process_year(config, 2024, index)
        assert result == 0

    @pytest.mark.asyncio
    async def test_streams_raw_and_optimises(self, tmp_path) -> None:
        """process_year streams to raw parquet then calls optimise_output."""
        config = AEFBNGConfig(
            years=[2024],
            bounds=(530_000, 180_000, 540_000, 190_000),
            output_path=str(tmp_path),
        )
        valid_array = np.ones((AEF_NUM_BANDS, 1000, 1000), dtype=np.int8)

        with (
            patch("aef_bng.pipeline.read_tile", new_callable=AsyncMock) as mock_read,
            patch("aef_bng.pipeline.reproject_tile_to_bng") as mock_reproject,
            patch("aef_bng.pipeline.optimise_output") as mock_optimise,
        ):
            from affine import Affine

            index = MagicMock()
            index.tiles_for_chunk.return_value = [
                {"path": "tile.tif", "crs": "EPSG:32630", "year": 2024}
            ]
            mock_read.return_value = (valid_array, Affine.identity(), "EPSG:32630")
            mock_reproject.return_value = valid_array

            result = await process_year(config, 2024, index)

        assert result == 1_000_000
        # optimise_output should have been called with raw path, output dir, and total rows
        mock_optimise.assert_called_once()
        call_args = mock_optimise.call_args
        assert str(call_args[0][0]).endswith("_raw.parquet")
        assert str(call_args[0][1]).endswith("2024")
        assert call_args[0][2] == 1_000_000


@pytest.mark.unit
class TestRunPipeline:
    """Tests for run_pipeline top-level orchestration."""

    @pytest.mark.asyncio
    async def test_returns_per_year_counts(self, tmp_path) -> None:
        """run_pipeline returns a dict mapping year → row count."""
        config = AEFBNGConfig(
            years=[2023, 2024],
            bounds=(530_000, 180_000, 540_000, 190_000),
            output_path=str(tmp_path),
        )

        with (
            patch("aef_bng.pipeline.AEFBNGIndex") as mock_index_cls,
            patch("aef_bng.pipeline.process_year", new_callable=AsyncMock) as mock_year,
        ):
            mock_index_cls.return_value = MagicMock()
            mock_year.side_effect = [500, 600]
            results = await run_pipeline(config)

        assert results == {2023: 500, 2024: 600}
        assert mock_year.call_count == 2
