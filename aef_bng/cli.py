"""CLI entry point for AEF-BNG processing."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import contextmanager

import click

from aef_bng.config import AEFBNGConfig
from aef_bng.constants import BNG_BOUNDS


@contextmanager
def stopwatch():
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()

    total_seconds = int(end_time - start_time)

    # Calculate HH:MM:SS
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
    click.secho(f"\nDuration: {timestamp}", fg="cyan", bold=True)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """AEF embeddings on British National Grid."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


@main.command()
@click.option(
    "--year",
    "-y",
    "years",
    type=int,
    multiple=True,
    required=True,
    help="Year(s) to process.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="./aef_bng_output",
    help="GeoParquet output directory.",
)
@click.option("--workers", "-w", type=int, default=4, help="Max concurrent workers.")
@click.option(
    "--bounds",
    nargs=4,
    type=int,
    default=BNG_BOUNDS,
    help="BNG bounds: minx miny maxx maxy.",
)
def process(
    years: tuple[int, ...],
    output_path: str,
    workers: int,
    bounds: tuple[int, int, int, int],
) -> None:
    """Process AEF embeddings to BNG GeoParquet files."""
    config = AEFBNGConfig(
        years=list(years),
        bounds=bounds,
        output_path=output_path,
        max_workers=workers,
    )

    with stopwatch():
        from aef_bng.pipeline import run_pipeline

        results = asyncio.run(run_pipeline(config))

    for year, rows in results.items():
        click.echo(f"Year {year}: {rows:,} rows written")


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option(
    "--png",
    "png_path",
    default="spatial_partitioning.png",
    show_default=True,
    help="Output path for the static PNG plot.",
)
@click.option(
    "--html",
    "html_path",
    default="spatial_partitioning.html",
    show_default=True,
    help="Output path for the interactive HTML map.",
)
def visualise(directory: str, png_path: str, html_path: str) -> None:
    """Visualise spatial partitioning of a GeoParquet dataset directory.

    DIRECTORY should contain .parquet files, e.g. london_aef/2025.

    Requires the viz extras:  uv sync --extra viz
    """
    from aef_bng.visualise import visualise as _visualise

    try:
        _visualise(directory, png_path=png_path, html_path=html_path)
    except ImportError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    click.secho(f"PNG  -> {png_path}", fg="green")
    click.secho(f"HTML -> {html_path}", fg="green")


@main.command(name="spark-run", hidden=True)
@click.option("--bounds", required=True)
@click.option("--years", required=True)
@click.option("--table-name", required=True)
@click.option("--resampling", default="nearest")
def spark_run(bounds: str, years: str, table_name: str, resampling: str) -> None:
    """Execute pipeline on-cluster (called by python_wheel_task)."""
    config = AEFBNGConfig(
        years=[int(y) for y in years.strip().strip(",").split(",")],
        bounds=tuple(int(b) for b in bounds.strip().strip(",").split(",")),  # type: ignore[arg-type]
        table_name=table_name,
        resampling=resampling,
    )

    from aef_bng.spark import process_with_spark

    process_with_spark(config)


def entrypoint() -> None:
    """Entry point for console script.

    Catches SystemExit(0) for Databricks python_wheel_task compatibility —
    Databricks treats any SystemExit as a task failure, but Click calls
    sys.exit(0) on success.
    """
    try:
        main()
    except SystemExit as e:
        if e.code != 0:
            raise


if __name__ == "__main__":
    main()
