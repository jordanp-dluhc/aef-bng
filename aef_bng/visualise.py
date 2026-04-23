"""Spatial partitioning visualiser for local GeoParquet datasets.

Produces two outputs from a directory of GeoParquet files:
  - A static matplotlib plot showing file and row-group extents
  - An interactive lonboard map saved as a standalone HTML file

Requires the ``viz`` optional dependencies::

    uv sync --extra viz
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

# Colours to distinguish files — mirrors the lonboard notebook palette
_COLORS = [
    "#FC49A3",  # pink
    "#CC66FF",  # purple-ish
    "#66CCFF",  # sky blue
    "#66FFCC",  # teal
    "#00FF00",  # lime green
    "#FFCC66",  # light orange
    "#FF6666",  # salmon
    "#FF0000",  # red
    "#FF8000",  # orange
    "#FFFF66",  # yellow
    "#00FFFF",  # turquoise
]

_VIZ_INSTALL_HINT = (
    "Visualisation dependencies are not installed.\nInstall them with:  uv sync --extra viz"
)


def _require_viz_deps() -> None:
    """Raise ImportError with install instructions if viz extras are missing."""
    missing = []
    for pkg, import_name in [
        ("geoarrow-rust-io", "geoarrow.rust.io"),
        ("lonboard", "lonboard"),
        ("matplotlib", "matplotlib"),
        ("contextily", "contextily"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(f"Missing viz dependencies: {', '.join(missing)}\n{_VIZ_INSTALL_HINT}")


def _hex_to_rgba(hex_color: str, alpha: int = 30) -> list[int]:
    """Convert a hex colour string to an [R, G, B, A] list."""
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


def _extract_bounds_from_fragment(
    fragment, directory: Path
) -> tuple[list[float] | None, list[list[float]]]:
    """Extract file and row-group bounds from a GeoParquetFile fragment.

    Tries geoarrow-rs first; falls back to geoparquet-io using the
    absolute file path.

    Args:
        fragment: A GeoParquetFile fragment from a GeoParquetDataset.
        directory: Absolute path to the dataset directory (for the fallback).

    Returns:
        Tuple of (file_bounds, rg_bounds) where each element is
        [xmin, ymin, xmax, ymax].
    """
    try:
        file_bounds = fragment.file_bbox()
        rg_structs = fragment.row_groups_bounds().to_pylist()
        rg_bounds = [[d["xmin"], d["ymin"], d["xmax"], d["ymax"]] for d in rg_structs]

        # file_bbox() returns None when the covering column uses a struct type
        # that geoarrow-rs can't read stats from. Derive from row groups instead.
        if file_bounds is None and rg_bounds:
            file_bounds = [
                min(rb[0] for rb in rg_bounds),
                min(rb[1] for rb in rg_bounds),
                max(rb[2] for rb in rg_bounds),
                max(rb[3] for rb in rg_bounds),
            ]

        return file_bounds, rg_bounds  # noqa: TRY300
    except Exception as e:
        import logging

        logging.getLogger(__name__).debug(
            "geoarrow-rs failed (%s), falling back to geoparquet-io", e
        )

    abs_path = str(directory / fragment.path)
    try:
        from geoparquet_io.core.duckdb_metadata import (
            find_primary_geometry_column_duckdb,
            get_aggregated_native_geo_stats,
            get_per_row_group_bbox_stats,
        )

        geom_col = find_primary_geometry_column_duckdb(abs_path)
        file_stats = get_aggregated_native_geo_stats(abs_path, geom_col)
        file_bounds = file_stats.get("bbox") if file_stats else None

        rg_stats = get_per_row_group_bbox_stats(abs_path, bbox_column=geom_col)
        rg_bounds = [
            [rg["xmin"], rg["ymin"], rg["xmax"], rg["ymax"]]
            for rg in rg_stats
            if all(k in rg for k in ("xmin", "ymin", "xmax", "ymax"))
        ]
        return file_bounds, rg_bounds  # noqa: TRY300
    except Exception as e2:
        raise RuntimeError(f"Both backends failed. geoparquet-io error: {e2}") from e2


def _static_plot(
    gdf_files: gpd.GeoDataFrame,
    per_file_rg_gdfs: list[gpd.GeoDataFrame],
    fragment_paths: list[str],
    output_path: str,
) -> None:
    """Render and save the static matplotlib figure.

    Left panel: file extents with basemap. Right panel: row-group extents
    coloured by file with basemap.

    Args:
        gdf_files: GeoDataFrame of file-level bounding boxes (EPSG:27700).
        per_file_rg_gdfs: One GeoDataFrame of row-group boxes per file.
        fragment_paths: File paths matching each entry in per_file_rg_gdfs.
        output_path: Where to save the PNG.
    """
    import contextily
    import geopandas as gpd
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from shapely.geometry import box

    n_rgs = sum(len(g) for g in per_file_rg_gdfs)
    total_polygon = gpd.GeoDataFrame(geometry=[box(*gdf_files.total_bounds)], crs=gdf_files.crs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: file extents
    total_polygon.plot(
        ax=ax1, facecolor="none", edgecolor="black", linewidth=2, linestyle="--", zorder=3
    )
    gdf_files.plot(
        ax=ax1, facecolor="steelblue", alpha=0.4, edgecolor="steelblue", linewidth=1, zorder=2
    )
    ax1.set_title(f"File Extents ({len(gdf_files)} files)", fontsize=14)
    ax1.set_xlabel("Easting")
    ax1.set_ylabel("Northing")
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.6, zorder=1)
    contextily.add_basemap(ax=ax1, crs=27700)

    # Right: row group extents coloured by file
    total_polygon.plot(
        ax=ax2, facecolor="none", edgecolor="black", linewidth=2, linestyle="--", zorder=3
    )
    rg_legend_patches = []
    for i, (gdf, path) in enumerate(zip(per_file_rg_gdfs, fragment_paths, strict=False)):
        if gdf.empty:
            continue
        color = _COLORS[i % len(_COLORS)]
        gdf.plot(
            ax=ax2,
            facecolor=color,
            alpha=0.3,
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
            linestyle="--",
        )
        rg_legend_patches.append(mpatches.Patch(facecolor=color, alpha=0.6, label=path))
    ax2.set_title(f"Row Group Extents ({n_rgs} row groups)", fontsize=14)
    ax2.set_xlabel("Easting")
    ax2.set_ylabel("Northing")
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle=":", alpha=0.6, zorder=1)
    contextily.add_basemap(ax=ax2, crs=27700)

    file_legend = [
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Total Extent"),
        mpatches.Patch(facecolor="steelblue", alpha=0.4, label="File Boundary"),
    ]
    fig.legend(
        handles=file_legend + rg_legend_patches,
        loc="lower center",
        ncol=min(len(file_legend) + len(rg_legend_patches), 4),
        bbox_to_anchor=(0.5, 0.02),
        fontsize=11,
    )

    plt.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _lonboard_map(
    gdf_files: gpd.GeoDataFrame,
    per_file_rg_gdfs: list[gpd.GeoDataFrame],
    fragment_paths: list[str],
    output_path: str,
) -> None:
    """Build and save an interactive lonboard map as a standalone HTML file.

    Layers:
      - File extents (blue, semi-transparent) — bottom
      - Row group extents per file (coloured by file) — on top

    Lonboard reprojects from EPSG:27700 to WGS84 automatically.

    Args:
        gdf_files: GeoDataFrame of file-level bounding boxes (EPSG:27700).
        per_file_rg_gdfs: One GeoDataFrame of row-group boxes per file.
        fragment_paths: File paths matching each entry in per_file_rg_gdfs.
        output_path: Where to save the HTML file.
    """
    from lonboard import Map, PolygonLayer  # type: ignore[import-not-found]
    from lonboard.basemap import CartoBasemap, MaplibreBasemap  # type: ignore[import-not-found]

    layers = []

    file_layer = PolygonLayer.from_geopandas(
        gdf_files,
        get_fill_color=[30, 144, 255, 25],
        get_line_color=[30, 144, 255, 200],
        line_width_min_pixels=1.5,
        auto_highlight=True,
    )
    layers.append(file_layer)

    for i, (gdf, _path) in enumerate(zip(per_file_rg_gdfs, fragment_paths, strict=False)):
        if gdf.empty:
            continue
        fill_rgba = _hex_to_rgba(_COLORS[i % len(_COLORS)], alpha=20)
        line_rgba = _hex_to_rgba(_COLORS[i % len(_COLORS)], alpha=180)
        layer = PolygonLayer.from_geopandas(
            gdf,
            get_fill_color=fill_rgba,
            get_line_color=line_rgba,
            line_width_min_pixels=0.6,
            auto_highlight=True,
        )
        layers.append(layer)

    m = Map(layers, basemap=MaplibreBasemap(style=CartoBasemap.DarkMatter), height=700)
    m.to_html(output_path, title="GeoParquet Spatial Partitioning")


def visualise(
    directory: str | Path,
    png_path: str = "spatial_partitioning.png",
    html_path: str = "spatial_partitioning.html",
) -> None:
    """Visualise spatial partitioning of a local GeoParquet dataset directory.

    Opens all ``.parquet`` files in ``directory`` as a ``GeoParquetDataset``,
    extracts file-level and row-group bounding boxes, then writes:

    - A static PNG with file extents and row-group extents on a basemap
    - An interactive HTML lonboard map coloured by file

    Args:
        directory: Directory containing ``.parquet`` files.
        png_path: Output path for the static PNG plot.
        html_path: Output path for the interactive HTML map.

    Raises:
        ImportError: If the ``viz`` optional dependencies are not installed.
        ValueError: If no ``.parquet`` files are found in ``directory``.
    """
    import logging

    _require_viz_deps()

    import geoarrow.rust.io as gio  # type: ignore[import-not-found]
    import geopandas as gpd
    from obstore.store import LocalStore
    from shapely.geometry import box

    logger = logging.getLogger(__name__)

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*reconstructed a store.*")

    store = LocalStore(str(directory))
    objects = [
        obj for obj in store.list_with_delimiter()["objects"] if obj["path"].endswith(".parquet")
    ]

    if not objects:
        raise ValueError(f"No .parquet files found in {directory}")

    dataset = gio.GeoParquetDataset.open(objects, store=store)
    logger.info(
        "Opened dataset: %d file(s), %d row groups, %s rows",
        len(dataset.fragments),
        dataset.num_row_groups,
        f"{dataset.num_rows:,}",
    )

    file_polygons: list = []
    per_file_rg_gdfs: list[gpd.GeoDataFrame] = []
    fragment_paths: list[str] = []

    for fragment in dataset.fragments:
        logger.debug("Processing %s", fragment.path)
        try:
            f_bounds, rg_bounds_list = _extract_bounds_from_fragment(fragment, directory)

            if f_bounds and len(f_bounds) >= 4:
                file_polygons.append(box(f_bounds[0], f_bounds[1], f_bounds[2], f_bounds[3]))

            rg_polys = [
                box(rb[0], rb[1], rb[2], rb[3]) for rb in rg_bounds_list if rb and len(rb) >= 4
            ]
            per_file_rg_gdfs.append(gpd.GeoDataFrame(geometry=rg_polys, crs="EPSG:27700"))
            fragment_paths.append(fragment.path)

        except Exception:
            logger.exception("Failed to process %s", fragment.path)

    if not file_polygons:
        raise ValueError("No bounds could be extracted from any file.")

    n_rgs = sum(len(g) for g in per_file_rg_gdfs)
    logger.info("Extracted bounds for %d file(s) and %d row groups", len(file_polygons), n_rgs)

    gdf_files = gpd.GeoDataFrame(geometry=file_polygons, crs="EPSG:27700")

    logger.info("Writing static plot -> %s", png_path)
    _static_plot(gdf_files, per_file_rg_gdfs, fragment_paths, png_path)

    logger.info("Writing lonboard map -> %s", html_path)
    _lonboard_map(gdf_files, per_file_rg_gdfs, fragment_paths, html_path)
