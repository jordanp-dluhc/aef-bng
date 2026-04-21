"""GeoParquet writer for AEF-BNG output.

Writes output as GeoParquet files using geoparquet-io for all spatial
optimisations:
- 10m BNG polygon geometry in EPSG:27700
- Bbox covering column (via geoparquet-io DuckDB spatial)
- Hilbert curve spatial sorting for optimal row-group locality
- KD-tree spatial partitioning for large datasets
- Zstd compression, ~100k record row groups
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path

import geoparquet_io as gpio
import numpy as np
import pyarrow as pa
from pyproj import CRS
from shapely import box as shapely_box
from shapely import to_wkb

from aef_bng.constants import BNG_RESOLUTION

logger = logging.getLogger(__name__)

# Row group size: 100k rows
ROW_GROUP_SIZE: int = 100_000

# GeoParquet distribution guidance: 1-2GB files, 50k-150k row groups.
# At ~60 bytes/row compressed for AEF BNG data: 1.5GB ≈ 25M rows.
# Use KD-tree partitioning only when total data would exceed a single ~1.5GB file.
_PARTITION_THRESHOLD: int = 25_000_000

# Target rows per KD-tree partition file (targeting ~1.5GB per file)
_TARGET_ROWS_PER_PARTITION: int = 25_000_000

# Cached CRS metadata (EPSG:27700 PROJJSON)
_CRS_META: dict | None = None


def _crs_metadata() -> dict:
    """Build minimal GeoParquet ``geo`` metadata with EPSG:27700 CRS (cached).

    geoparquet-io reads the CRS from this metadata when writing the file.
    The bbox covering reference is added automatically by ``add_bbox()``.
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


def write_geoparquet(tables: list[pa.Table], output_dir: str) -> int:
    """Write Arrow tables as optimised GeoParquet files.

    Concatenates tables, builds 10m BNG polygon geometry, then delegates all
    spatial optimisation to geoparquet-io:

    - ``add_bbox()`` — adds a bbox struct covering column via DuckDB spatial
    - ``sort_hilbert()`` — reorders rows by Hilbert curve for spatial locality
    - ``partition_by_kdtree()`` — splits into spatially-balanced files for
      large datasets; ``write()`` is used for datasets under
      ``_ROWS_PER_PARTITION`` rows.

    Input tables must have ``easting`` and ``northing`` int32 columns
    (lower-left corner of each 10m cell). These are replaced by WKB polygon
    geometry before the geoparquet-io pipeline runs.

    Args:
        tables: Arrow tables with columns including easting, northing.
        output_dir: Output directory for the part files.

    Returns:
        Total number of rows written.
    """
    non_empty = [t for t in tables if t.num_rows > 0]
    if not non_empty:
        return 0

    combined = pa.concat_tables(non_empty)
    total_rows = combined.num_rows

    # Build WKB polygon geometry from easting/northing coordinates
    eastings = combined.column("easting").to_numpy().astype(np.float64)
    northings = combined.column("northing").to_numpy().astype(np.float64)
    geom_wkb = _build_wkb_column(eastings, northings)

    # Replace coordinate columns with WKB geometry
    output = combined.drop_columns(["easting", "northing"])
    output = output.append_column("geometry", geom_wkb)

    # Attach CRS metadata so geoparquet-io picks up EPSG:27700
    geo_json = json.dumps(_crs_metadata()).encode()
    output = output.replace_schema_metadata({b"geo": geo_json})

    # geoparquet-io pipeline: bbox → Hilbert sort → write
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    table = gpio.Table(output, geometry_column="geometry").add_bbox().sort_hilbert()

    if total_rows >= _PARTITION_THRESHOLD:
        # Spatial KD-tree partitioning for large datasets
        iterations = _kdtree_iterations(total_rows)
        table.partition_by_kdtree(
            str(out_path),
            iterations=iterations,
            hive=False,
            compression="ZSTD",
            compression_level=15,
        )
        logger.info(
            "GeoParquet write complete: %d rows, %d KD-tree partitions -> %s",
            total_rows,
            2**iterations,
            output_dir,
        )
    else:
        part_file = str(out_path / f"part-{uuid.uuid4().hex[:12]}.parquet")
        table.write(
            part_file,
            row_group_rows=ROW_GROUP_SIZE,
            compression="ZSTD",
            geoparquet_version="1.1",
        )
        logger.info("GeoParquet write complete: %d rows -> %s", total_rows, part_file)

    return total_rows


def _build_wkb_column(
    eastings: np.ndarray,
    northings: np.ndarray,
) -> pa.Array:
    """Build a WKB binary array of 10m x 10m BNG cell polygons.

    Vectorised via shapely: avoids per-row Python overhead for millions of cells.

    Args:
        eastings: Lower-left easting of each cell (float64).
        northings: Lower-left northing of each cell (float64).

    Returns:
        PyArrow binary array of WKB polygons.
    """
    res = float(BNG_RESOLUTION)
    geoms = shapely_box(eastings, northings, eastings + res, northings + res)
    return pa.array(to_wkb(geoms), type=pa.binary())


def _kdtree_iterations(total_rows: int) -> int:
    """Compute KD-tree iterations to target ~``_TARGET_ROWS_PER_PARTITION`` rows per file.

    Creates ``2^iterations`` spatially-balanced partitions. The target is
    ~1.5M rows per file (~90MB at typical compression), which geoparquet-io
    recommends for efficient spatial filter pushdown.

    Args:
        total_rows: Total number of rows to partition.

    Returns:
        Number of KD-tree split iterations, in [1, 9].
    """
    return max(1, min(9, math.ceil(math.log2(total_rows / _TARGET_ROWS_PER_PARTITION))))
