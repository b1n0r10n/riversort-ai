# Dataset pipeline

The dataset CLI converts an immutable source export into a validated,
group-aware YOLO snapshot. It operates only on a data root outside the
repository and does not access the internet.

```bash
export RIVERSORT_DATA_ROOT=../RiverSort_AI_Data
python scripts/data_pipeline/cli.py --help
```

Available stages cover inventory, safe extraction, FPWR staging, audit queue
generation, duplicate/sequence grouping, splitting, promotion, and validation.
Writing commands support dry-run and refuse to overwrite raw data or an
existing snapshot.

See [DATA_AND_MODEL.md](../../docs/DATA_AND_MODEL.md) for the active source,
class mapping, quality controls, and snapshot identifiers.
