# Notices and attribution

RiverSort AI includes or depends on the following open-source work and data.

## Ultralytics YOLO

The detector and tracker integration uses
[Ultralytics](https://github.com/ultralytics/ultralytics), distributed under
the GNU Affero General Public License v3.0. RiverSort AI and its published
fine-tuned weight are distributed under AGPL-3.0-or-later.

## Float Plastic Waste River

The published baseline was trained with Float Plastic Waste River v6:

- project: `float-plastic-waste-river`;
- creator/workspace: `floatplasticwasteriver`;
- dataset author attribution shown by the project: Antonius Didi Kurniadi;
- source: <https://universe.roboflow.com/floatplasticwasteriver/float-plastic-waste-river/dataset/6>;
- license: Creative Commons Attribution 4.0 International (CC BY 4.0);
- source export: 2,824 images and 5,761 annotations;
- modifications: class remapping, bounding-box clamping in staging,
  duplicate/sequence grouping, and a deterministic group-aware split.

The dataset images are not redistributed in this repository. Users who obtain
the dataset must follow the source terms and retain attribution.

## Project visuals

The dashboard screenshot and SVG diagrams under `docs/assets/` were produced
from the RiverSort AI application or from project data. They do not contain
third-party test media.
