# Spark Pipeline: AEF Embeddings to BNG Delta Table

How the pipeline distributes work across a Databricks cluster and writes to a Unity Catalog Delta table.

---

## Entry point

`process_with_spark(config)` in `spark.py` is the Spark entrypoint. The relevant `AEFBNGConfig` fields are:

| Field | Purpose |
|-------|---------|
| `years` | List of years to process |
| `bounds` | BNG bounding box (EPSG:27700 metres) |
| `table_name` | Unity Catalog three-part name, e.g. `my_catalog.schema.aef_bng` |
| `resampling` | Resampling method (default `nearest`) |
| `chunk_size` | Processing unit size in metres (default 10,000m) |

---

## Step 1 — Index and config broadcasts

Two values are loaded on the driver and broadcast to all executors:

1. **AEF tile index** — the STAC GeoParquet index is loaded with predicate pushdown (only tiles overlapping the requested bounds and years), pickled, and broadcast. Executors deserialise it once per partition, not per row.
2. **Resampling method** — the resampling string from config.

---

## Step 2 — Building the work queue

`_build_chunks_dataframe` creates a Spark DataFrame where **each row is one `(10km BNG chunk, year)` pair**.

The UK is divided into 10km BNG grid squares (e.g. `TQ38`). Each square is 1,000×1,000 pixels at 10m resolution. These are crossed with all requested years to produce the full task list.

The DataFrame schema is **deliberately flat** — bounding boxes are split into four separate `IntegerType` columns rather than a struct or array. This is required because `mapInArrow` works with Arrow `RecordBatch` and nested types are awkward to extract per-row in Python.

The DataFrame is repartitioned to `min(num_tasks, defaultParallelism × 2)`. With 80 tasks this keeps one task per chunk-year, avoiding partitions larger than one row.

---

## Step 3 — `mapInArrow` processing

`mapInArrow` is Spark's Arrow-native UDF. It hands each partition to a Python function as an iterator of `pa.RecordBatch`, bypassing JVM serialisation overhead.

For each row in the batch, `_process_chunk_row` runs:

### 3a. Spatial query

`index.tiles_for_chunk(chunk, year)` returns the AEF COG tile paths (and their UTM CRS) that spatially overlap this 10km BNG square for the requested year. Most UK chunks have 1–2 tiles; chunks near the UTM 30/31 zone boundary (roughly 0° longitude) may have 3.

### 3b. Async S3 reads (`reader.py`)

For each tile, `read_tile` issues windowed HTTP range requests via `obstore` + `async-geotiff`. The window is the chunk's BNG bounds reprojected into the tile's UTM CRS with a 500m padding buffer. Only the internal COG tiles overlapping that window are fetched — typically 2–10 MB per source tile.

The async calls run inside `asyncio.run(...)` — a fresh event loop per chunk-year row. Tiles within a chunk are read **sequentially** (not with `asyncio.gather`), so the async benefit is within a single tile's chunked HTTP reads, not across tiles.

### 3c. Reproject (`reproject.py`)

`reproject_tile_to_bng` calls `rasterio.warp.reproject` to warp the source tile from its UTM CRS into BNG at 10m, producing a `(64, 1000, 1000)` int8 array. This is the **CPU-intensive** step (GDAL warp of 64 bands × 1M pixels).

### 3d. Merge (`reproject.py`)

`merge_tiles` applies a first-valid strategy across any overlapping tiles. The first tile contributes its value for each pixel; subsequent tiles only fill where all 64 bands are nodata (`-128`). Embeddings are treated as atomic 64-band vectors — partial replacement is never done.

### 3e. Extract (`extract.py`)

`extract_pixels_spark` flattens the merged `(64, 1000, 1000)` array into an Arrow table with columns `bng_ref`, `year`, `A00`..`A63` (64 individual `int8` columns), and `
_wkb`. Nodata pixels are dropped.

The `geometry_wkb` column contains standard WKB polygons for each 10m cell, built directly in numpy (no shapely). Each polygon is a fixed 93-byte little-endian WKB: a 13-byte Polygon header followed by 5 float64 coordinate pairs closing the rectangle.

---

## Step 4 — Geometry conversion

After `mapInArrow`, the `geometry_wkb` binary column is converted to a Databricks geometry type:

```python
from pyspark.databricks.sql import functions as dbf

result_df = result_df.withColumn(
    "geometry", dbf.st_geomfromwkb(F.col("geometry_wkb"), F.lit(27700))
).drop("geometry_wkb")
```

This is applied as a single Spark SQL operation on the full DataFrame before writing — no per-row Python processing.

---

## Step 5 — Write to Unity Catalog

```python
result_df.write.format("delta")
    .mode("append")
    .option("overwriteSchema", "true")
    .saveAsTable(table_name)  # three-part Unity Catalog name

spark.sql(f"ALTER TABLE {table_name} CLUSTER BY (year, bng_ref)")
```

- **Append mode** — safe to re-run for additional years or sub-regions. Note: re-running the same years/bounds will produce duplicate rows; the table has no deduplication by default.
- **`saveAsTable`** with a three-part name registers the table in Unity Catalog automatically. No manual `CREATE TABLE` is needed.
- **Liquid clustering on `(year, bng_ref)`** is applied after writing so queries filtering by year and then joining on BNG reference are co-located on disk.

### Output schema

| Column | Type | Notes |
|--------|------|-------|
| `bng_ref` | `STRING` | 10-char 10m BNG reference, e.g. `TQ30008000` |
| `year` | `SMALLINT` | AEF year |
| `A00`..`A63` | `TINYINT` | 64 individual int8 embedding bands |
| `geometry` | `GEOMETRY` | 10m × 10m BNG cell polygon (EPSG:27700) |

Both local (GeoParquet) and Spark (Delta) modes always include geometry. Local mode writes WKB polygons in the `geometry` column; Spark mode uses the Databricks `GEOMETRY` type via `st_geomfromwkb`.

---

## Cluster configuration and recommendations

### Reference spec

| Setting | Value |
|---------|-------|
| DBR | 17.3 LTS (Spark 3.5) |
| Node type | Standard_DS3 (28 GB RAM, 8 vCPUs) |
| Min / max workers | 1 / 10 |
| `spark.task.cpu` | 1 |
| Max concurrent tasks | 80 (8 cores × 10 workers) |

### Is this spec suitable?

**Yes, broadly.** Memory is generous — a `(64, 1000, 1000)` int8 array is 64 MB; with Arrow overhead and rasterio working buffers, each task uses well under 1 GB. At 3.5 GB per task slot (28 GB ÷ 8 tasks) there is substantial headroom.

80 tasks is a reasonable degree of parallelism for this workload.

### The GDAL threading issue

The CPU-heavy operation is `rasterio.warp.reproject` (Step 3c). GDAL, which backs rasterio, can spawn its own internal threads controlled by the `GDAL_NUM_THREADS` environment variable. **If this is left at its default (often `ALL_CPUS`), 8 concurrent Spark tasks on a node may each try to use all 8 cores, creating 64 threads competing for 8 cores.**

**Recommended fix:** set `GDAL_NUM_THREADS=1` in the Databricks cluster environment variables:

```
GDAL_NUM_THREADS=1
```

This matches the `spark.task.cpu=1` allocation exactly. With 8 tasks per node, each task's GDAL uses 1 thread → 8 threads on 8 cores, no oversubscription.

### Does async-geotiff need multiple CPU cores?

No. `async-geotiff` + `obstore` use Python's `asyncio` for S3 HTTP reads. `asyncio` is **single-threaded cooperative multitasking** — it yields control back to the event loop while waiting for S3 responses (pure I/O wait), not by spawning threads. It does not benefit from additional CPU cores.

The read phase is **I/O bound**, not CPU bound. The CPU is largely idle during S3 reads, which is why `spark.task.cpu=1` is appropriate. The CPU spike comes from `rasterio.warp.reproject` immediately after the reads complete.

### `spark.task.cpu=1` vs `spark.task.cpu=2` trade-off

| Setting | Tasks per node | GDAL threads available | Memory per task |
|---------|----------------|------------------------|-----------------|
| `spark.task.cpu=1` + `GDAL_NUM_THREADS=1` | 8 | 1 | 3.5 GB |
| `spark.task.cpu=2` + `GDAL_NUM_THREADS=2` | 4 | 2 | 7 GB |

These must be set in matching pairs — mismatching them (e.g. `spark.task.cpu=1` with `GDAL_NUM_THREADS=2`) oversubscribes cores.

**Recommendation: keep `spark.task.cpu=1` and set `GDAL_NUM_THREADS=1`.** The resampling method is `nearest`, which is memory-bandwidth-bound. GDAL warp threads don't speed it up meaningfully. Maximising Spark parallelism (80 tasks) is more valuable than giving GDAL extra threads per task.

### Other recommendations

**1. Use spot/preemptible instances for workers.**
Each task processes one chunk independently and failures are retried by Spark automatically. The pipeline is well-suited to spot instances, which can reduce cost significantly.

**2. Watch for duplicate rows on re-run.**
The write uses `mode("append")`. Re-running the same `years`/`bounds` will append duplicate rows. Consider tracking which years have been written (e.g. via a checkpoint table) or adding a deduplication step downstream.

**3. `spark.sql.shuffle.partitions`.**
The pipeline doesn't shuffle, so this is less relevant. The default (200) is fine.

**4. Liquid clustering runs after the write.**
`ALTER TABLE ... CLUSTER BY` triggers an asynchronous background optimise job in Databricks. Queries issued immediately after the write may not yet benefit from clustering. Run `OPTIMIZE table_name` explicitly if you need clustering before the first query.
