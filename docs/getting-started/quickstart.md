# Quick Start

## On Databricks

```python
from aef_bng.spark import process_with_spark
from aef_bng.config import AEFBNGConfig

config = AEFBNGConfig(
    years=[2025],
    bounds=(0, 0, 700000, 1300000),  # Full Great Britain
    table_name="catalog.schema.aef_embeddings_bng",
)

process_with_spark(config)
```

## CLI (via Databricks python_wheel_task)

```bash
aef-bng spark-run \
  --bounds "0,0,700000,1300000" \
  --years "2025" \
  --table-name "catalog.schema.aef_embeddings_bng"
```
