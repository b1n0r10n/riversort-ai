# Model pipeline

The model CLI records training and evaluation against a versioned dataset
snapshot.

```bash
export RIVERSORT_DATA_ROOT=../RiverSort_AI_Data
python scripts/model_pipeline/cli.py --help
```

Experiment output belongs under the external data root and must not be
committed. The repository stores only configuration, checksums, the model card,
and compact numeric reports.

The published baseline configuration is
`configs/training_yolo11n_baseline.yaml`. Test metrics and limitations are
documented in [MODEL_CARD.md](../../models/MODEL_CARD.md).
