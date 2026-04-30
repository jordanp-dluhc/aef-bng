"""GeoParquet writer for AEF-BNG output.

Two-phase write strategy:
  1. Stream: PyArrow ParquetWriter appends chunks with O(row_group) memory
  2. Optimise: gpio CLI (DuckDB-backed) sorts + partitions with GeoParquet 2.0

This enables processing arbitrarily large extents without OOM.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import uuid
from pathlib import Path

import geopandas as gpd
import geoparquet_io as gpio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS

from aef_bng.constants import AEF_BAND_NAMES, BNG_RESOLUTION

logger = logging.getLogger(__name__)

# Row group size for the streaming writer (controls flush frequency / memory)
_STREAMING_ROW_GROUP_SIZE: int = 100_000

# Row group size for final optimised output
_OPTIMISED_ROW_GROUP_SIZE: int = 100_000

# Target rows per output file (~1 GB compressed at ~67 bytes/row)
_TARGET_ROWS_PER_FILE: int = 15_000_000

# Cached CRS metadata (EPSG:27700 PROJJSON)
_CRS_META: dict | None = None


def _crs_metadata() -> dict:
    """Build minimal GeoParquet ``geo`` metadata with EPSG:27700 CRS (cached).

    geoparquet-io reads the CRS from this metadata when writing the file.
    """
    global _CRS_META
    if _CRS_META is None:
        _CRS_META = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Polygon"],
                    "crs": CRS.from_epsg(27700).to_json_dict(),
                }
            },
        }
    return _CRS_META


def _output_schema() -> pa.Schema:
    """Build the Arrow schema for the streaming raw parquet file.

    Includes bng_ref, year, 64 band columns, easting, northing, and WKB geometry.
    """
    fields: list[pa.Field] = [
        pa.field("bng_ref", pa.string()),
        pa.field("year", pa.int16()),
    ]
    fields.extend(pa.field(name, pa.int8()) for name in AEF_BAND_NAMES)
    fields.append(pa.field("easting", pa.int32()))
    fields.append(pa.field("northing", pa.int32()))
    fields.append(pa.field("geometry", pa.binary()))

    schema = pa.schema(fields)
    geo_json = json.dumps(_crs_metadata()).encode()
    return schema.with_metadata({b"geo": geo_json})


def _build_wkb_column(
    eastings: np.ndarray,
    northings: np.ndarray,
) -> pa.Array:
    """Build a WKB binary array of 10m x 10m BNG cell polygons.

    Constructed entirely in numpy — no shapely, no per-row Python loops.
    Each polygon is a fixed 93-byte little-endian WKB:
    - 13-byte header (byte order, geometry type, ring count, point count)
    - 5 x 2 x float64 coordinate pairs closing the rectangle

    Args:
        eastings: Lower-left easting of each cell (float64).
        northings: Lower-left northing of each cell (float64).

    Returns:
        PyArrow binary array of WKB polygons.
    """
    n = len(eastings)
    e = eastings.astype(np.float64)
    n_ = northings.astype(np.float64)
    res = float(BNG_RESOLUTION)
    e1 = e + res
    n1 = n_ + res

    # 5 coordinates: LL, LR, UR, UL, LL (closed ring)
    coords = np.empty((n, 10), dtype=np.float64)
    coords[:, 0] = e
    coords[:, 1] = n_
    coords[:, 2] = e1
    coords[:, 3] = n_
    coords[:, 4] = e1
    coords[:, 5] = n1
    coords[:, 6] = e
    coords[:, 7] = n1
    coords[:, 8] = e
    coords[:, 9] = n_

    # WKB header: byte_order=1 (LE), type=3 (Polygon), num_rings=1, num_points=5
    header = np.frombuffer(
        b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00",
        dtype=np.uint8,
    )
    wkb = np.empty((n, 93), dtype=np.uint8)
    wkb[:, :13] = header
    wkb[:, 13:] = coords.view(np.uint8).reshape(n, 80)

    flat = wkb.tobytes()
    offsets = np.arange(0, (n + 1) * 93, 93, dtype=np.int32)
    return pa.Array.from_buffers(
        pa.binary(),
        n,
        [None, pa.py_buffer(offsets.tobytes()), pa.py_buffer(flat)],
    )


def prepare_table_for_write(table: pa.Table) -> pa.Table:
    """Convert a pipeline output table to GeoParquet-ready format.

    Appends a WKB geometry column built from easting/northing and attaches
    GeoParquet schema metadata. Easting and northing are retained as integer
    columns for direct use in spatial queries and visualisation.

    Args:
        table: Arrow table with easting and northing int32 columns.

    Returns:
        Arrow table with easting, northing, and geometry columns plus geo metadata.
    """
    eastings = table.column("easting").to_numpy().astype(np.float64)
    northings = table.column("northing").to_numpy().astype(np.float64)
    geom_wkb = _build_wkb_column(eastings, northings)

    output = table.append_column("geometry", geom_wkb)

    geo_json = json.dumps(_crs_metadata()).encode()
    return output.replace_schema_metadata({b"geo": geo_json})


class StreamingParquetWriter:
    """Streams Arrow tables to a raw GeoParquet file with O(row_group) memory.

    Each call to ``write_table()`` converts easting/northing to WKB geometry
    and appends to the parquet file. The writer flushes row groups at
    ``row_group_size`` intervals.

    Usage::

        writer = StreamingParquetWriter(path)
        for table in chunk_tables:
            writer.write_table(table)
        writer.close()
    """

    def __init__(self, path: Path, row_group_size: int = _STREAMING_ROW_GROUP_SIZE) -> None:
        self._path = path
        self._row_group_size = row_group_size
        self._schema = _output_schema()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            str(path),
            schema=self._schema,
            compression="ZSTD",
        )
        self._total_rows = 0

    @property
    def total_rows(self) -> int:
        """Total rows written so far."""
        return self._total_rows

    @property
    def path(self) -> Path:
        """Path to the raw parquet file."""
        return self._path

    def write_table(self, table: pa.Table) -> None:
        """Append a chunk table (with easting/northing) to the raw parquet file.

        Converts coordinates to WKB geometry before writing.

        Args:
            table: Arrow table from extract_pixels() with easting/northing columns.
        """
        geo_table = prepare_table_for_write(table)
        self._writer.write_table(geo_table, row_group_size=self._row_group_size)
        self._total_rows += table.num_rows

    def close(self) -> None:
        """Close the parquet writer, finalizing the file."""
        self._writer.close()


def _compute_partitions(total_rows: int) -> int:
    """Compute the KD-tree partition count (power of 2) targeting ~15M rows/file.

    Examples:
        50M rows  → 4 partitions  (12.5M/file)
        91M rows  → 8 partitions  (11.4M/file)
        195M rows → 16 partitions (12.2M/file)
        10M rows  → 1 partition   (no split needed)

    Args:
        total_rows: Total number of rows in the dataset.

    Returns:
        Partition count as a power of 2 (minimum 1).
    """
    import math

    if total_rows <= _TARGET_ROWS_PER_FILE:
        return 1

    ideal = total_rows / _TARGET_ROWS_PER_FILE
    exponent = round(math.log2(ideal))
    return max(1, 2**exponent)


def optimise_output(raw_path: Path, output_dir: Path, total_rows: int) -> None:
    """Sort and partition a raw GeoParquet file using the gpio CLI.

    Two-step optimization using DuckDB-backed operations (O(1) memory):
      1. Hilbert sort with GeoParquet 2.0
      2. KD-tree partition into evenly-distributed files targeting ~15M rows each

    The partition count is computed as the nearest power of 2 to
    ``total_rows / 15M``, ensuring uniform row distribution across files.

    Args:
        raw_path: Path to the unsorted raw parquet file.
        output_dir: Final output directory for partitioned files.
        total_rows: Total number of rows (used to compute partition count).

    Raises:
        RuntimeError: If either gpio command fails.
    """
    sorted_path = raw_path.with_suffix(".sorted.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpio_bin = _find_gpio()

    # Step 1: Hilbert sort → GeoParquet 2.0
    logger.info("Hilbert sorting -> %s", sorted_path.name)
    _run_gpio(
        gpio_bin,
        "sort",
        "hilbert",
        str(raw_path),
        str(sorted_path),
        "--geoparquet-version",
        "2.0",
        "--row-group-size",
        str(_OPTIMISED_ROW_GROUP_SIZE),
        "--add-bbox",
    )

    # Step 2: KD-tree partition with explicit partition count for uniform distribution
    partitions = _compute_partitions(total_rows)

    if partitions <= 1:
        # No partitioning needed — just move the sorted file to output
        final_path = output_dir / sorted_path.name
        sorted_path.rename(final_path)
        logger.info("Single file (no partition needed): %d rows -> %s", total_rows, final_path)
    else:
        rows_per_file = total_rows // partitions
        logger.info(
            "KD-tree partitioning: %d rows -> %d files (~%d rows/file) -> %s",
            total_rows,
            partitions,
            rows_per_file,
            output_dir,
        )
        _run_gpio(
            gpio_bin,
            "partition",
            "kdtree",
            str(sorted_path),
            str(output_dir),
            "--partitions",
            str(partitions),
            "--geoparquet-version",
            "2.0",
            "--row-group-size",
            str(_OPTIMISED_ROW_GROUP_SIZE),
            "--overwrite",
        )
        sorted_path.unlink(missing_ok=True)

    # Cleanup raw temp file
    raw_path.unlink(missing_ok=True)
    logger.info("Optimization complete: %s", output_dir)


def _find_gpio() -> str:
    """Locate the gpio CLI binary, preferring the current venv."""
    venv_gpio = Path(sys.executable).parent / "gpio"
    if venv_gpio.exists():
        return str(venv_gpio)
    return "gpio"


def _run_gpio(gpio_bin: str, *args: str) -> None:
    """Run a gpio CLI command, raising RuntimeError on failure."""
    cmd = [gpio_bin, *args]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(
            f"gpio command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Output reader
# ---------------------------------------------------------------------------


def read_output(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read AEF-BNG GeoParquet output files into a GeoDataFrame.

    ``gpd.read_parquet`` requires GeoParquet 1.x metadata (a ``"geo"`` key in
    the Parquet file metadata).  The pipeline writes GeoParquet 2.0 via the
    ``gpio`` CLI, which uses GeoArrow extension-type encoding instead, so
    geopandas raises ``ValueError: Missing geo metadata``.

    This function uses ``geoparquet_io`` to read the file (DuckDB-backed, so
    it handles GeoParquet 2.0 natively), then extracts WKB bytes via
    ``to_pylist()`` and reconstructs the GeoDataFrame with
    ``GeoSeries.from_wkb``.

    When ``bbox`` is provided it is pushed down to the parquet reader as a
    pyarrow filter expression on the ``bbox`` struct column
    (``xmin``/``ymin``/``xmax``/``ymax``) added by the ``--add-bbox`` flag
    during optimisation.  This avoids loading rows outside the query window.

    Args:
        path: Path to a parquet file or directory of parquet files.
        bbox: Optional ``(minx, miny, maxx, maxy)`` spatial filter in
            EPSG:27700 (BNG metres).
        columns: Optional list of columns to load.  ``geometry`` is always
            added to the request regardless of this list.

    Returns:
        GeoDataFrame with EPSG:27700 polygon geometry.
    """
    import pyarrow.compute as pc

    load_cols = columns
    if load_cols is not None and "geometry" not in load_cols:
        load_cols = [*load_cols, "geometry"]

    filters = None
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        filters = (
            (pc.field("bbox", "xmax") >= minx)
            & (pc.field("bbox", "ymax") >= miny)
            & (pc.field("bbox", "xmin") <= maxx)
            & (pc.field("bbox", "ymin") <= maxy)
        )

    table = gpio.read(str(path), columns=load_cols, filters=filters).to_arrow()

    geom_bytes = table.column("geometry").to_pylist()
    geom_series = gpd.GeoSeries.from_wkb(geom_bytes, crs=27700)

    keep = [n for n in table.schema.names if n != "geometry"]
    df = table.select(keep).to_pandas()

    return gpd.GeoDataFrame(df, geometry=geom_series)


# ---------------------------------------------------------------------------
# Legacy API (retained for backwards compatibility with tests)
# ---------------------------------------------------------------------------


def write_geoparquet(tables: list[pa.Table], output_dir: str) -> int:
    """Write Arrow tables as optimised GeoParquet files (legacy API).

    Concatenates tables, builds geometry, sorts, and writes. Kept for
    backwards compatibility with existing tests.

    Args:
        tables: Arrow tables with columns including easting, northing.
        output_dir: Output directory for the part files.

    Returns:
        Total number of rows written.
    """
    import geoparquet_io as gpio

    non_empty = [t for t in tables if t.num_rows > 0]
    if not non_empty:
        return 0

    combined = pa.concat_tables(non_empty)
    total_rows = combined.num_rows

    output = prepare_table_for_write(combined)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    table = gpio.Table(output, geometry_column="geometry").add_bbox().sort_hilbert()

    part_file = str(out_path / f"part-{uuid.uuid4().hex[:12]}.parquet")
    table.write(
        part_file,
        row_group_rows=_STREAMING_ROW_GROUP_SIZE,
        compression="ZSTD",
        geoparquet_version="2.0",
    )
    logger.info("GeoParquet write complete: %d rows -> %s", total_rows, part_file)

    return total_rows
