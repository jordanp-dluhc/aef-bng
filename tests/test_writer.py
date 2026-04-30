"""Tests for aef_bng.writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aef_bng.constants import AEF_NUM_BANDS, BNG_RESOLUTION
from aef_bng.extract import extract_pixels
from aef_bng.grid import ChunkSpec
from aef_bng.writer import (
    StreamingParquetWriter,
    _build_wkb_column,
    _compute_partitions,
    prepare_table_for_write,
    write_geoparquet,
)


def _make_test_table(n: int = 10) -> pa.Table:
    """Create a small test table using extract_pixels."""
    chunk = ChunkSpec(
        bng_10km_ref="TQ38",
        bounds_bng=(
            530_000,
            180_000,
            530_000 + n * BNG_RESOLUTION,
            180_000 + n * BNG_RESOLUTION,
        ),
        bounds_wgs84=(-0.15, 51.48, -0.14, 51.49),
        shape=(n, n),
    )
    data = np.ones((AEF_NUM_BANDS, n, n), dtype=np.int8)
    return extract_pixels(data, chunk, 2024)


@pytest.mark.unit
class TestComputePartitions:
    """Tests for _compute_partitions."""

    def test_small_dataset_no_partition(self) -> None:
        """Dataset below target returns 1 (no split)."""
        assert _compute_partitions(10_000_000) == 1

    def test_at_target_no_partition(self) -> None:
        """Dataset at exactly target returns 1."""
        assert _compute_partitions(15_000_000) == 1

    def test_50m_rows(self) -> None:
        """50M rows → 4 partitions (12.5M/file)."""
        assert _compute_partitions(50_000_000) == 4

    def test_91m_rows(self) -> None:
        """91M rows (all GB) → 8 partitions (11.4M/file)."""
        assert _compute_partitions(91_000_000) == 8

    def test_195m_rows(self) -> None:
        """195M rows → 16 partitions (12.2M/file)."""
        assert _compute_partitions(194_954_498) == 16

    def test_always_power_of_two(self) -> None:
        """Result is always a power of 2."""
        import math

        for n in [30_000_000, 60_000_000, 120_000_000, 200_000_000]:
            result = _compute_partitions(n)
            assert result >= 1
            assert math.log2(result) == int(math.log2(result))


@pytest.mark.unit
class TestBuildWkbColumn:
    """Tests for _build_wkb_column."""

    def test_returns_binary_array(self) -> None:
        """Result is a PyArrow binary array."""
        result = _build_wkb_column(np.array([530_000.0]), np.array([180_000.0]))
        assert result.type == pa.binary()

    def test_length_matches_input(self) -> None:
        """One WKB entry per input coordinate."""
        result = _build_wkb_column(
            np.array([530_000.0, 531_000.0, 532_000.0]),
            np.array([180_000.0, 181_000.0, 182_000.0]),
        )
        assert len(result) == 3

    def test_decodes_to_polygon(self) -> None:
        """WKB decodes to a 10m x 10m Polygon at the correct location."""
        from shapely.wkb import loads as wkb_loads

        result = _build_wkb_column(np.array([530_000.0]), np.array([180_000.0]))
        poly = wkb_loads(result[0].as_py())
        assert poly.geom_type == "Polygon"
        assert abs(poly.area - 100.0) < 1e-6
        assert poly.bounds == (530_000.0, 180_000.0, 530_010.0, 180_010.0)


@pytest.mark.unit
class TestPrepareTableForWrite:
    """Tests for prepare_table_for_write."""

    def test_adds_geometry_retains_coordinates(self) -> None:
        """Geometry column is added; easting and northing are preserved."""
        table = _make_test_table(5)
        result = prepare_table_for_write(table)
        assert "easting" in result.column_names
        assert "northing" in result.column_names
        assert "geometry" in result.column_names

    def test_attaches_geo_metadata(self) -> None:
        """Output table has GeoParquet 'geo' schema metadata."""
        table = _make_test_table(5)
        result = prepare_table_for_write(table)
        meta = result.schema.metadata
        assert b"geo" in meta
        geo = json.loads(meta[b"geo"])
        assert geo["primary_column"] == "geometry"

    def test_preserves_row_count(self) -> None:
        """Row count is unchanged."""
        table = _make_test_table(10)
        result = prepare_table_for_write(table)
        assert result.num_rows == 100


@pytest.mark.unit
class TestStreamingParquetWriter:
    """Tests for StreamingParquetWriter."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Writer creates a parquet file."""
        path = tmp_path / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))
        writer.close()
        assert path.exists()

    def test_total_rows_accumulates(self, tmp_path: Path) -> None:
        """total_rows tracks cumulative rows written."""
        path = tmp_path / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))  # 25 rows
        writer.write_table(_make_test_table(10))  # 100 rows
        writer.close()
        assert writer.total_rows == 125

    def test_file_contains_all_rows(self, tmp_path: Path) -> None:
        """Parquet file contains all rows from multiple writes."""
        path = tmp_path / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))  # 25
        writer.write_table(_make_test_table(10))  # 100
        writer.close()

        result = pq.read_table(str(path))
        assert result.num_rows == 125

    def test_file_has_geometry_and_coordinate_columns(self, tmp_path: Path) -> None:
        """Output file has WKB geometry column alongside easting and northing."""
        path = tmp_path / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))
        writer.close()

        result = pq.read_table(str(path))
        assert "geometry" in result.column_names
        assert "easting" in result.column_names
        assert "northing" in result.column_names

    def test_file_has_geo_metadata(self, tmp_path: Path) -> None:
        """Output file has GeoParquet metadata in the schema."""
        path = tmp_path / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))
        writer.close()

        schema = pq.read_schema(str(path))
        assert b"geo" in schema.metadata
        geo = json.loads(schema.metadata[b"geo"])
        assert geo["primary_column"] == "geometry"

    def test_geometries_are_valid(self, tmp_path: Path) -> None:
        """Written geometries decode to valid 10m polygons."""
        from shapely.wkb import loads as wkb_loads

        path = tmp_path / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))
        writer.close()

        result = pq.read_table(str(path))
        wkb_col = result.column("geometry")
        for i in range(min(5, len(wkb_col))):
            poly = wkb_loads(wkb_col[i].as_py())
            assert poly.geom_type == "Polygon"
            assert abs(poly.area - 100.0) < 1e-6

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Writer creates parent directories if they don't exist."""
        path = tmp_path / "nested" / "dir" / "test.parquet"
        writer = StreamingParquetWriter(path)
        writer.write_table(_make_test_table(5))
        writer.close()
        assert path.exists()


@pytest.mark.unit
class TestWriteGeoparquetLegacy:
    """Tests for the legacy write_geoparquet API."""

    def test_empty_input_returns_zero(self, tmp_path: Path) -> None:
        """Empty table list writes nothing and returns 0."""
        result = write_geoparquet([], str(tmp_path / "out"))
        assert result == 0

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """At least one parquet file is created in the output directory."""
        table = _make_test_table(10)
        out_dir = str(tmp_path / "output")
        write_geoparquet([table], out_dir)
        parquet_files = list(Path(out_dir).glob("**/*.parquet"))
        assert len(parquet_files) >= 1

    def test_returns_correct_row_count(self, tmp_path: Path) -> None:
        """Returned row count matches total input rows."""
        table = _make_test_table(10)  # 10x10 = 100 rows
        result = write_geoparquet([table], str(tmp_path / "out"))
        assert result == 100

    def test_geoparquet_metadata_present(self, tmp_path: Path) -> None:
        """Output file contains valid GeoParquet 'geo' schema metadata."""
        table = _make_test_table(5)
        out_dir = str(tmp_path / "out")
        write_geoparquet([table], out_dir)
        parquet_file = next(Path(out_dir).glob("**/*.parquet"))
        schema_meta = pq.read_schema(str(parquet_file)).metadata
        assert b"geo" in schema_meta
        geo = json.loads(schema_meta[b"geo"])
        assert geo["primary_column"] == "geometry"

    def test_geometries_are_valid_polygons(self, tmp_path: Path) -> None:
        """Written geometries decode to valid 10m x 10m BNG cell polygons."""
        from shapely.wkb import loads as wkb_loads

        table = _make_test_table(10)
        out_dir = str(tmp_path / "out")
        write_geoparquet([table], out_dir)
        parquet_file = next(Path(out_dir).glob("**/*.parquet"))
        result = pq.read_table(str(parquet_file))
        wkb_col = result.column("geometry")
        for i in range(min(10, len(wkb_col))):
            poly = wkb_loads(wkb_col[i].as_py())
            assert poly.geom_type == "Polygon"
            assert abs(poly.area - 100.0) < 1e-6
