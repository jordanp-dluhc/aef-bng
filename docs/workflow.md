# Workflow

## Pipeline Overview

The pipeline processes AEF satellite embeddings for Great Britain in parallel
10km BNG chunks via Spark `mapInArrow`.

```
Input: AEF COGs on S3 (UTM projection, 64 bands, 10m resolution)
       ↓
[1] STAC Index Query — find overlapping UTM tiles for each BNG chunk
       ↓
[2] Async COG Read — fetch tile data via obstore + async-geotiff
       ↓
[3] Reproject — UTM → BNG (EPSG:27700), first-valid merge at zone boundaries
       ↓
[4] Extract — vectorised pixel extraction, WKB geometry per pixel
       ↓
Output: Delta table (chip_id, 64 band columns, geometry)
```

## Databricks Cluster Configuration

| Setting | Value |
|---------|-------|
| Runtime | DBR ML 17.3 LTS |
| Workers | Standard_D4_v2 (CPU) |
| `spark.task.cpus` | 1 |

The pipeline is CPU-bound (no GPU required) — reprojection and pixel
extraction are the bottlenecks.

## Output Schema

Each row represents a single 10m pixel:

| Column | Type | Description |
|--------|------|-------------|
| bng_ref | string | 10m BNG grid reference |
| band_00..band_63 | int8 | 64 embedding dimensions |
| geometry | binary (WKB) | Point geometry in EPSG:27700 |
| year | int | Source data year |
