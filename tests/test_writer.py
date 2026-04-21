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
    _TARGET_ROWS_PER_PARTITION,
    _build_wkb_column,
    _kdtree_iterations,
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
class TestKdtreeIterations:
    """Tests for KD-tree iteration calculation."""

    def test_at_target_returns_one(self) -> None:
        """Dataset at exactly _TARGET_ROWS_PER_PARTITION returns 1 iteration (2 partitions)."""
        assert _kdtree_iterations(_TARGET_ROWS_PER_PARTITION) == 1

    def test_two_times_target(self) -> None:
        """2x target → 1 iteration (2 partitions, ~1 file each)."""
        assert _kdtree_iterations(_TARGET_ROWS_PER_PARTITION * 2) == 1

    def test_ten_times_target(self) -> None:
        """10x target → 4 iterations (16 partitions)."""
        assert _kdtree_iterations(_TARGET_ROWS_PER_PARTITION * 10) == 4  # ceil(log2(10)) = 4

    def test_capped_at_nine(self) -> None:
        """Iterations are capped at 9 regardless of dataset size."""
        assert _kdtree_iterations(10**12) == 9


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
class TestWriteGeoparquet:
    """Tests for write_geoparquet."""

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

    def test_multiple_tables_combined(self, tmp_path: Path) -> None:
        """Multiple tables are combined and all rows written."""
        t1 = _make_test_table(5)  # 25 rows
        t2 = _make_test_table(10)  # 100 rows
        result = write_geoparquet([t1, t2], str(tmp_path / "out"))
        assert result == 125

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

    def test_geometry_type_is_polygon(self, tmp_path: Path) -> None:
        """geometry column is declared as Polygon in GeoParquet metadata."""
        table = _make_test_table(5)
        out_dir = str(tmp_path / "out")
        write_geoparquet([table], out_dir)
        parquet_file = next(Path(out_dir).glob("**/*.parquet"))
        schema_meta = pq.read_schema(str(parquet_file)).metadata
        geo = json.loads(schema_meta[b"geo"])
        assert "Polygon" in geo["columns"]["geometry"]["geometry_types"]

    def test_bbox_covering_column_present(self, tmp_path: Path) -> None:
        """Output file contains a bbox struct covering column."""
        table = _make_test_table(5)
        out_dir = str(tmp_path / "out")
        write_geoparquet([table], out_dir)
        parquet_file = next(Path(out_dir).glob("**/*.parquet"))
        schema = pq.read_schema(str(parquet_file))
        assert "bbox" in schema.names

    def test_output_columns(self, tmp_path: Path) -> None:
        """Output has bng_ref, year, bands, geometry, bbox; no easting/northing."""
        table = _make_test_table(5)
        out_dir = str(tmp_path / "out")
        write_geoparquet([table], out_dir)
        parquet_file = next(Path(out_dir).glob("**/*.parquet"))
        col_names = pq.read_schema(str(parquet_file)).names
        assert "bng_ref" in col_names
        assert "year" in col_names
        assert "geometry" in col_names
        assert "bbox" in col_names
        assert "easting" not in col_names
        assert "northing" not in col_names

    def test_row_count_in_file_matches_input(self, tmp_path: Path) -> None:
        """Total rows in all written files match the input table row count."""
        table = _make_test_table(10)  # 100 rows
        out_dir = str(tmp_path / "out")
        write_geoparquet([table], out_dir)
        total = sum(pq.read_metadata(str(f)).num_rows for f in Path(out_dir).glob("**/*.parquet"))
        assert total == 100

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
