"""Local processing orchestration for AEF-BNG pipeline.

Two-phase architecture:
  1. Stream: process chunks and append to a raw parquet file (O(row_group) memory)
  2. Optimise: gpio CLI sorts (Hilbert) and partitions (KD-tree) with GeoParquet 2.0

This enables processing arbitrarily large extents without OOM while producing
optimally-partitioned output for spatial queries.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from aef_bng.constants import AEF_NODATA
from aef_bng.extract import extract_pixels
from aef_bng.grid import BNGOutputGrid, ChunkSpec
from aef_bng.index import AEFBNGIndex
from aef_bng.reader import bng_bounds_to_utm, read_tile
from aef_bng.reproject import merge_tiles, reproject_tile_to_bng
from aef_bng.writer import StreamingParquetWriter, optimise_output

if TYPE_CHECKING:
    import pyarrow as pa

    from aef_bng.config import AEFBNGConfig

logger = logging.getLogger(__name__)


async def process_chunk(
    chunk: ChunkSpec,
    year: int,
    index: AEFBNGIndex,
    config: AEFBNGConfig,
) -> pa.Table | None:
    """Process a single 10km BNG chunk for a given year.

    Reads overlapping AEF tiles, reprojects them to BNG, merges them,
    and extracts pixel data as an Arrow table.

    Args:
        chunk: The BNG chunk specification.
        year: Year to process.
        index: Loaded AEF tile index.
        config: Pipeline configuration.

    Returns:
        Arrow table of extracted pixels, or None if no data.
    """
    tiles = index.tiles_for_chunk(chunk, year)
    if not tiles:
        return None

    dst_transform = chunk.transform
    if dst_transform is None:
        raise RuntimeError(f"ChunkSpec {chunk.bng_10km_ref} has no transform")

    # read and reproject each tile (windowed to chunk extent)
    reprojected: list[np.ndarray] = []
    for tile in tiles:
        try:
            tile_crs = str(tile["crs"])
            window_bounds = bng_bounds_to_utm(chunk.bounds_bng, tile_crs)
            tile_data = await read_tile(str(tile["path"]), window_bounds=window_bounds)
            if tile_data is None:
                continue
            src_data, src_tf, src_crs = tile_data
            result = reproject_tile_to_bng(
                src_data,
                src_tf,
                src_crs,
                dst_transform,
                chunk.shape,
                resampling=config.resampling,
            )
            reprojected.append(result)
        except Exception:
            logger.exception(
                "Failed to read/reproject tile %s for chunk %s",
                tile["path"],
                chunk.bng_10km_ref,
            )

    if not reprojected:
        return None

    # merge overlapping tiles (first-valid strategy)
    merged = merge_tiles(reprojected)

    # check if entire chunk is nodata
    if np.all(merged == AEF_NODATA):
        return None

    return extract_pixels(merged, chunk, year)


async def process_year(config: AEFBNGConfig, year: int, index: AEFBNGIndex) -> int:
    """Process all chunks for a single year.

    Phase 1: Streams chunk tables to a raw temp parquet file.
            Up to ``config.max_workers`` chunks are read and reprojected concurrently;
            results are written to disk as each task completes.
    Phase 2: Optimises output with gpio (Hilbert sort + KD-tree partition + GeoParquet 2.0).

    Args:
        config: Pipeline configuration.
        year: Year to process.
        index: Pre-loaded AEF tile index.

    Returns:
        Total number of rows written.
    """
    grid = BNGOutputGrid(config.bounds, config.chunk_size)
    chunks = list(grid.enumerate_chunks())

    semaphore = asyncio.Semaphore(config.max_workers)
    output_dir = Path(config.output_path) / str(year)
    raw_path = output_dir / "_raw.parquet"

    async def _bounded(chunk: ChunkSpec) -> pa.Table | None:
        async with semaphore:
            return await process_chunk(chunk, year, index, config)

    # Phase 1: Stream chunks to raw parquet (concurrent reads, sequential writes)
    writer = StreamingParquetWriter(raw_path)

    tasks = [asyncio.create_task(_bounded(c)) for c in chunks]
    with tqdm(total=len(tasks), desc=f"Year {year}", unit="chunk") as pbar:
        for coro in asyncio.as_completed(tasks):
            table = await coro
            if table is not None and table.num_rows > 0:
                writer.write_table(table)
            pbar.update(1)

    writer.close()
    total_rows = writer.total_rows

    if total_rows == 0:
        raw_path.unlink(missing_ok=True)
        logger.info("Year %d: no data found", year)
        return 0

    # Phase 2: Optimise with gpio (Hilbert sort + KD-tree + GeoParquet 2.0)
    logger.info("Year %d: %d rows written, optimizing output...", year, total_rows)
    optimise_output(raw_path, output_dir, total_rows)
    logger.info("Year %d complete: %d total rows -> %s", year, total_rows, output_dir)

    return total_rows


async def run_pipeline(config: AEFBNGConfig) -> dict[int, int]:
    """Run the full pipeline for all configured years.

    Args:
        config: Pipeline configuration.

    Returns:
        Dict mapping year -> total rows written.
    """
    # load index once for all years
    index = AEFBNGIndex()
    index.load_for_bounds(config.bounds, config.years)

    results: dict[int, int] = {}
    for year in config.years:
        logger.info("Processing year %d", year)
        results[year] = await process_year(config, year, index)
        logger.info("Year %d: %d rows", year, results[year])

    return results
