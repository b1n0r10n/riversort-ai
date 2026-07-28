# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local Streamlit viewer/editor for the FPWR audit queue."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st
from PIL import Image, ImageDraw

from riversort.data_pipeline import TARGET_NAMES, parse_yolo_label, read_csv, resolve_data_root, write_csv


root = resolve_data_root()
staging = root / "staging/fpwr4"
audit_path = root / "manifests/fpwr4_audit/audit_log.csv"
rows = read_csv(audit_path)
manifest = {item["asset_id"]: item for item in read_csv(staging / "file_manifest.csv")}


def annotated(image_path, label_path, class_map, target: bool):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in parse_yolo_label(label_path, class_map):
        x1, y1, x2, y2 = box.xyxy
        draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), outline="red", width=3)
        class_id = box.target_class_id if target else box.source_class_id
        draw.text((x1 * image.width, max(0, y1 * image.height - 12)), str(class_id), fill="yellow")
    return image

st.set_page_config(page_title="FPWR audit", layout="wide")
st.title("FPWR four-class manual audit")
index = st.number_input("Queue index", min_value=0, max_value=max(len(rows) - 1, 0), value=0)
row = rows[int(index)]
asset = manifest[row["asset_id"]]
raw = root / "raw/source_exports/fpwr_v6"
left, right = st.columns(2)
left.image(
    annotated(raw / asset["source_image"], raw / asset["source_label"], {0: 0, 1: 1, 2: 2, 3: 3}, False),
    caption="Source bbox/class",
    width="stretch",
)
right.image(
    annotated(staging / "images" / asset["image_name"], staging / "labels" / asset["label_name"], {0: 0, 1: 1, 2: 2, 3: 3}, True),
    caption="Staging bbox/target class",
    width="stretch",
)
st.json(row)

with st.form("decision"):
    auditor = st.text_input("Auditor", value=row["auditor"] or "")
    audit_status = st.selectbox("Audit status", ["REVIEW", "ACCEPTED", "REJECTED"], index=["REVIEW", "ACCEPTED", "REJECTED"].index(row["audit_status"]))
    bbox_status = st.selectbox("BBox status", ["REVIEW", "ACCEPTED", "REJECTED"], index=["REVIEW", "ACCEPTED", "REJECTED"].index(row["bbox_status"]))
    class_status = st.selectbox("Class status", ["REVIEW", "ACCEPTED", "REJECTED"], index=["REVIEW", "ACCEPTED", "REJECTED"].index(row["class_status"]))
    scene_flags = st.text_input("Scene flags", value=row["scene_flags"])
    reason = st.text_area("Reason", value=row["reason"])
    if st.form_submit_button("Save decision"):
        row.update(
            auditor=auditor,
            audit_status=audit_status,
            bbox_status=bbox_status,
            class_status=class_status,
            scene_flags=scene_flags,
            reason=reason,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
        write_csv(audit_path, rows)
        st.success("Saved")
