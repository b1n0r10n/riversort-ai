# Model card: RiverSort YOLO11n FPWR4

## Model details

| Field | Value |
|---|---|
| Model version | `0.1.0-baseline.1` |
| Architecture | Ultralytics YOLO11n detection |
| Capability profile | `riversort_plastic4_mvp` `1.0.0` |
| Training snapshot | `riversort_fpwr4-20260722-r2` |
| Snapshot SHA-256 | `74badb41175aacef724f6b2c4db722ce46dcd6875546f9035472d60e46690b2c` |
| Weight SHA-256 | `cc235703c6b8b9f6244e68145ca82fea115c95526b91e5716468c3f2d2c31235` |
| Runtime confidence | `0.40` |
| Runtime IoU | `0.50` |
| Runtime image size | `960` |
| License | `AGPL-3.0-or-later` |

## Intended use

The model supports RiverSort AI image and short-video analysis for four
floating plastic categories:

```text
0 plastic_bottle
1 plastic_bag
2 plastic_packaging_sachet
3 other_plastic
```

Target class `4 other_floating_debris` is not part of the model. Applications
must represent it as `not_supported` with null metrics.

The model is intended for prototype decision support and visual documentation.
It is not a physical measurement instrument, a cleanup controller, or a model
validated for national-scale generalization.

## Training data

The model was trained on Float Plastic Waste River v6, distributed under
CC BY 4.0:

<https://universe.roboflow.com/floatplasticwasteriver/float-plastic-waste-river/dataset/6>

Snapshot summary:

- 2,824 images;
- 5,761 annotations;
- 1,947 train images;
- 437 validation images;
- 440 test images;
- group-aware split, seed 42;
- no known group leakage;
- no dedicated negative samples.

Source labels were mapped to the RiverSort class order. Bounding boxes outside
normalized bounds were clamped in staging with a correction log. Raw source
files were not modified.

## Training procedure

- 100 epochs with patience 20;
- batch size 8;
- training image size 640;
- deterministic seed 42;
- AMP enabled;
- image cache disabled;
- online augmentation on train only.

The detector confidence was selected on validation from a grid of 0.05–0.50
using macro-F1. The selected value, 0.40, was frozen before test evaluation.

## Held-out FPWR evaluation

Matching uses IoU 0.50 and greedy one-to-one class matching.

| Class | Precision | Recall | F1 | AP50 |
|---|---:|---:|---:|---:|
| `plastic_bottle` | 0.8535 | 0.8481 | 0.8508 | 0.8802 |
| `plastic_bag` | 0.7778 | 0.6488 | 0.7074 | 0.7113 |
| `plastic_packaging_sachet` | 0.7249 | 0.7959 | 0.7588 | 0.8156 |
| `other_plastic` | 0.5625 | 0.3673 | 0.4444 | 0.3675 |

Summary:

- mAP50: `0.6936`;
- macro-F1: `0.6904`;
- false positives per image: `0.4227`;
- missed objects per image: `0.5273`.

Batch-1 detector latency on an NVIDIA GeForce RTX 4050 Laptop GPU was
6.48 ms median and 6.76 ms p90 after 20 warmup images over 100 test images.
This is a hardware-specific detector benchmark, not end-to-end application
latency.

## Limitations

- Training data comes from one source.
- Dedicated negative samples are absent.
- Small, distant, occluded, and low-contrast objects are frequently missed.
- Dense scenes increase missed detections and track fragmentation.
- `other_plastic` has substantially lower recall than the other classes.
- Reflections, vegetation, and unfamiliar objects can produce false positives.
- Nonplastic floating debris is unsupported.
- Results have not been validated for nationwide deployment.

An empty result must not be interpreted as a clean river. Video counts should
be treated as indicative when quality flags report unstable tracking or class
votes.

## Ethical and operational considerations

Operators must have permission to process uploaded media and should avoid
collecting faces, vehicle plates, or sensitive location metadata without a
lawful purpose. Cleanup decisions require human verification.

The appropriate claim for model output is:

> Estimasi komposisi visual berdasarkan jumlah objek unik dan proporsi area
> deteksi.

The output does not estimate weight, mass, tonnage, or physical volume.

## Distribution

Download and verify the model with:

```bash
python scripts/download_model.py
```

The weight is a GitHub Release asset rather than a tracked Git file. The
downloader verifies the SHA-256 value and class map before use.
