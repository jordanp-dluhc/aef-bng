"""Spark/Databricks distributed processing for AEF-BNG.

Distributes 10km chunk processing across a Spark cluster using
``mapInArrow`` for Arrow-native processing, writing output to a
Unity Catalog Delta table with liquid clustering.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from aef_bng.constants import AEF_BAND_NAMES, AEF_NODATA
from aef_bng.extract import extract_pixels_spark
from aef_bng.grid import BNGOutputGrid, ChunkSpec
from aef_bng.index import AEFBNGIndex
from aef_bng.reader import bng_bounds_to_utm, read_tile
from aef_bng.reproject import merge_tiles, reproject_tile_to_bng

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyspark.sql import DataFrame, SparkSession  # type: ignore  # noqa: PGH003

    from aef_bng.config import AEFBNGConfig

logger = logging.getLogger(__name__)


def _output_schema() -> pa.Schema:
    """Build the mapInArrow output schema.

    Returns:
        PyArrow schema for the UDF output, including the ``geometry_wkb`` field.
    """
    fields: list[tuple[str, pa.DataType]] = (
        [("bng_ref", pa.string()), ("year", pa.int16())]
        + [(name, pa.int8()) for name in AEF_BAND_NAMES]
        + [("geometry_wkb", pa.binary())]
    )
    return pa.schema(fields)


def _process_chunk_row(
    bng_10km_ref: str,
    bounds_bng: tuple[int, int, int, int],
    bounds_wgs84: tuple[float, float, float, float],
    year: int,
    index: AEFBNGIndex,
    resampling: str,
) -> pa.Table | None:
    """Process a single chunk from flat column values.

    Args:
        bng_10km_ref: 10km BNG grid reference.
        bounds_bng: BNG bounding box (minx, miny, maxx, maxy).
        bounds_wgs84: WGS84 bounding box (minx, miny, maxx, maxy).
        year: Year to process.
        index: Loaded AEF tile index.
        resampling: Resampling method name.

    Returns:
        Arrow table with (bng_ref, year, embedding, geometry_wkb) or None if no data.
    """
    chunk = ChunkSpec(
        bng_10km_ref=bng_10km_ref,
        bounds_bng=bounds_bng,
        bounds_wgs84=bounds_wgs84,
    )

    tiles = index.tiles_for_chunk(chunk, year)
    if not tiles:
        return None

    async def _read_and_reproject() -> np.ndarray | None:
        reprojected = []
        for tile in tiles:
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
                chunk.transform,  # type: ignore[arg-type]
                chunk.shape,
                resampling=resampling,
            )
            reprojected.append(result)
        if not reprojected:
            return None
        return merge_tiles(reprojected)

    merged = asyncio.run(_read_and_reproject())
    if merged is None or np.all(merged == AEF_NODATA):
        return None

    return extract_pixels_spark(merged, chunk, year)


def _make_process_partition(
    broadcast_index_bytes: Any,
    broadcast_resampling: Any,
) -> Any:
    """Create the partition processing function with broadcast references.

    Returns the function to pass to ``mapInArrow``. The broadcasts are dereferenced once per
    partition (not per row), amortising deserialisation.

    Args:
        broadcast_index_bytes: Spark broadcast of pickled AEFBNGIndex.
        broadcast_resampling: Spark broadcast of resampling method string.

    Returns:
        Callable suitable for ``mapInArrow``.
    """

    def process_partition(batch_iter: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Process all chunks in a partition, yielding output batches.

        Deserialises the index once per partition then iterates over rows in each input RecordBatch.
        """
        index: AEFBNGIndex = pickle.loads(broadcast_index_bytes.value)  # noqa: S301
        resampling: str = broadcast_resampling.value
        schema = _output_schema()

        for batch in batch_iter:
            bng_refs = batch.column("bng_10km_ref").to_pylist()
            bounds_bng_0 = batch.column("bounds_bng_0").to_pylist()
            bounds_bng_1 = batch.column("bounds_bng_1").to_pylist()
            bounds_bng_2 = batch.column("bounds_bng_2").to_pylist()
            bounds_bng_3 = batch.column("bounds_bng_3").to_pylist()
            bounds_wgs84_0 = batch.column("bounds_wgs84_0").to_pylist()
            bounds_wgs84_1 = batch.column("bounds_wgs84_1").to_pylist()
            bounds_wgs84_2 = batch.column("bounds_wgs84_2").to_pylist()
            bounds_wgs84_3 = batch.column("bounds_wgs84_3").to_pylist()
            years = batch.column("year").to_pylist()

            output_tables: list[pa.Table] = []

            for i in range(batch.num_rows):
                try:
                    result = _process_chunk_row(
                        bng_10km_ref=bng_refs[i],
                        bounds_bng=(
                            bounds_bng_0[i],
                            bounds_bng_1[i],
                            bounds_bng_2[i],
                            bounds_bng_3[i],
                        ),
                        bounds_wgs84=(
                            bounds_wgs84_0[i],
                            bounds_wgs84_1[i],
                            bounds_wgs84_2[i],
                            bounds_wgs84_3[i],
                        ),
                        year=years[i],
                        index=index,
                        resampling=resampling,
                    )
                    if result is not None and result.num_rows > 0:
                        output_tables.append(result)
                except Exception:
                    logger.exception(
                        "Failed to process chunk %s year %d",
                        bng_refs[i],
                        years[i],
                    )

            if output_tables:
                combined = pa.concat_tables(output_tables)
                yield combined.to_batches()[0] if combined.num_rows > 0 else _empty_batch(schema)
            else:
                yield _empty_batch(schema)

    return process_partition


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    """Return an empty RecordBatch matching the given schema."""
    columns = {field.name: pa.array([], type=field.type) for field in schema}
    return pa.RecordBatch.from_pydict(columns, schema=schema)


def _build_chunks_dataframe(
    spark: SparkSession,
    config: AEFBNGConfig,
) -> DataFrame:
    """Build a Spark DataFrame with one row per (chunk, year) combination.

    Uses flat columns for Arrow compatibility — no nested structs or arrays.

    Args:
        spark: Active SparkSession.
        config: Pipeline configuration.

    Returns:
        Spark DataFrame with the flat input schema.
    """
    from pyspark.sql.types import (  # type: ignore  # noqa: PGH003
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    grid = BNGOutputGrid(config.bounds, config.chunk_size)
    chunks = grid.enumerate_chunks()

    rows = []
    for chunk in chunks:
        for year in config.years:
            rows.append(
                (
                    chunk.bng_10km_ref,
                    chunk.bounds_bng[0],
                    chunk.bounds_bng[1],
                    chunk.bounds_bng[2],
                    chunk.bounds_bng[3],
                    chunk.bounds_wgs84[0],
                    chunk.bounds_wgs84[1],
                    chunk.bounds_wgs84[2],
                    chunk.bounds_wgs84[3],
                    year,
                )
            )

    input_schema = StructType(
        [
            StructField("bng_10km_ref", StringType(), False),
            StructField("bounds_bng_0", IntegerType(), False),
            StructField("bounds_bng_1", IntegerType(), False),
            StructField("bounds_bng_2", IntegerType(), False),
            StructField("bounds_bng_3", IntegerType(), False),
            StructField("bounds_wgs84_0", DoubleType(), False),
            StructField("bounds_wgs84_1", DoubleType(), False),
            StructField("bounds_wgs84_2", DoubleType(), False),
            StructField("bounds_wgs84_3", DoubleType(), False),
            StructField("year", IntegerType(), False),
        ]
    )

    return spark.createDataFrame(rows, schema=input_schema)


def process_with_spark(config: AEFBNGConfig) -> None:
    """Run the AEF-BNG pipeline distributed across a Spark cluster.

    Creates a DataFrame of chunk specs with flat columns, broadcasts the
    AEF index, and processes partitions in parallel using ``mapInArrow``.
    Writes results to a Unity Catalog Delta table with liquid clustering.

    Args:
        config: Pipeline configuration.
    """
    from pyspark.sql import SparkSession  # type: ignore  # noqa: PGH003

    spark = SparkSession.builder.getOrCreate()

    index = AEFBNGIndex()
    index.load_for_bounds(config.bounds, config.years)
    index_bytes = pickle.dumps(index)
    broadcast_index = spark.sparkContext.broadcast(index_bytes)
    broadcast_resampling = spark.sparkContext.broadcast(config.resampling)

    chunks_df = _build_chunks_dataframe(spark, config)

    grid = BNGOutputGrid(config.bounds, config.chunk_size)
    num_tasks = len(grid.enumerate_chunks()) * len(config.years)
    num_partitions = min(num_tasks, spark.sparkContext.defaultParallelism * 2)
    chunks_df = chunks_df.repartition(num_partitions)

    schema = _output_schema()
    process_fn = _make_process_partition(broadcast_index, broadcast_resampling)
    result_df = chunks_df.mapInArrow(process_fn, schema=schema)

    from pyspark.databricks.sql import functions as dbf  # type: ignore[import-not-found]
    from pyspark.sql import functions as F  # type: ignore  # noqa: PGH003

    result_df = result_df.withColumn(
        "geometry", dbf.st_geomfromwkb(F.col("geometry_wkb"), F.lit(27700))
    ).drop("geometry_wkb")

    table_name = config.table_name
    if table_name:
        (
            result_df.write.format("delta")
            .mode("append")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
        spark.sql(f"ALTER TABLE {table_name} CLUSTER BY (year, bng_ref)")
        logger.info("Spark processing complete. Output table: %s", table_name)
    else:
        (result_df.write.format("delta").mode("append").save(config.output_path))
        logger.info("Spark processing complete. Output at %s", config.output_path)
