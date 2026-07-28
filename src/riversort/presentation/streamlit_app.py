# Copyright (C) 2026 RiverSort AI contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local/offline Streamlit dashboard for RiverSort AI."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from riversort.application import AnalysisBundle, AnalysisError, UploadedMedia
from riversort.domain import QualityFlagCode, SupportStatus
from riversort.infrastructure import load_project_config
from riversort.infrastructure.config import capability_profile
from riversort.infrastructure.runtime import RuntimeServices, build_runtime_services


LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _load_services(config_dir: str) -> RuntimeServices:
    return build_runtime_services(load_project_config(config_dir))


def _clear_result() -> None:
    st.session_state.pop("analysis_bundle", None)
    st.session_state.pop("analysis_mode", None)
    st.session_state.pop("analysis_input_key", None)


def _progress_writer(status: Any):
    labels = {
        "validation": "Memvalidasi media",
        "normalization": "Menyesuaikan resolusi video",
        "inference": "Menjalankan object detection",
        "tracking": "Memperbarui unique track",
        "aggregation": "Menghitung komposisi dan prioritas",
        "encoding": "Membuat preview dengan label track final",
        "export": "Membuat preview dan export",
    }

    def update(stage: str) -> None:
        status.write(labels.get(stage, stage))

    return update


def _render_capability(config) -> None:
    profile = capability_profile(config)
    st.sidebar.header("Kemampuan model")
    st.sidebar.success("4 kelas plastik aktif")
    for class_name in profile.supported_classes.values():
        st.sidebar.markdown(f"- `{class_name}`")
    for class_name in profile.unsupported_target_classes.values():
        st.sidebar.warning(f"`{class_name}`: not_supported")
    st.sidebar.caption(
        "Model belum menilai debris nonplastik; indeks dapat lebih rendah dari kondisi lapangan."
    )


def _render_composition(bundle: AnalysisBundle) -> None:
    supported = [
        item
        for item in bundle.result.compositions
        if item.support_status is SupportStatus.SUPPORTED
    ]
    rows = [
        {
            "Kelas": item.class_name,
            "Jumlah objek unik": item.object_count or 0,
            "Komposisi jumlah (%)": (
                round(item.object_count_share * 100, 2)
                if item.object_count_share is not None
                else None
            ),
            "Proporsi area deteksi (%)": (
                round(item.detected_area_share * 100, 2)
                if item.detected_area_share is not None
                else None
            ),
        }
        for item in supported
    ]
    frame = pd.DataFrame(rows)
    left, right = st.columns(2)
    with left:
        st.subheader("Komposisi berdasarkan jumlah objek unik")
        st.caption("Pembagi hanya empat kelas yang didukung model.")
        if frame["Komposisi jumlah (%)"].notna().any():
            chart = px.bar(
                frame,
                x="Kelas",
                y="Komposisi jumlah (%)",
                text="Komposisi jumlah (%)",
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Tidak ada objek valid untuk menghitung komposisi jumlah.")
    with right:
        st.subheader("Komposisi berdasarkan proporsi area deteksi")
        st.caption("Bounding box bukan segmentasi atau ukuran fisik sampah.")
        if frame["Proporsi area deteksi (%)"].notna().any():
            chart = px.bar(
                frame,
                x="Kelas",
                y="Proporsi area deteksi (%)",
                text="Proporsi area deteksi (%)",
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Tidak ada area valid untuk menghitung komposisi area.")
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_priority(bundle: AnalysisBundle) -> None:
    priority = bundle.result.priority
    st.subheader("Indeks prioritas pembersihan")
    if priority.status == "not_assessable":
        st.warning("Tidak dapat dinilai — tidak ada objek valid dari kelas yang didukung.")
    else:
        score, tier = st.columns(2)
        score.metric("Skor visual", f"{priority.score}/100")
        tier.metric("Tier", str(priority.tier).upper())
    st.caption(
        f"Scope: `{priority.scope}` · config `{priority.config_version}`. "
        "Residual override untuk debris nonplastik tidak aktif pada profile "
        "empat kelas. Ini indeks visual prototype, bukan ukuran bahaya fisik absolut."
    )
    if priority.contributions:
        contributions = pd.DataFrame(
            [
                {"Komponen": key, "Nilai": value}
                for key, value in priority.contributions.items()
            ]
        )
        with st.expander("Lihat komponen perhitungan"):
            st.dataframe(contributions, use_container_width=True, hide_index=True)


def _render_recommendations(bundle: AnalysisBundle) -> None:
    st.subheader("Rekomendasi berbasis aturan")
    if _requires_manual_review(bundle):
        st.warning(
            "Rekomendasi ini bersifat indikatif karena kualitas klasifikasi atau "
            "tracking memerlukan verifikasi manual."
        )
    if not bundle.result.recommendations:
        st.info("Tidak ada rule yang aktif untuk hasil ini.")
        return
    for item in bundle.result.recommendations:
        with st.expander(f"{item.rule_id} — {item.action}", expanded=True):
            st.write(item.reason)
            st.code(
                f"{item.observed_metric}: {item.observed_value} "
                f"{item.operator} {item.threshold}",
                language=None,
            )
            st.caption(
                f"Rule config {item.rule_version}; evidence: "
                + ", ".join(item.evidence_fields)
            )


def _render_result(bundle: AnalysisBundle) -> None:
    result = bundle.result
    st.divider()
    st.header("Hasil analisis")
    if result.media_info and result.media_info.resolution_normalized:
        st.info(
            "Resolusi diperkecil otomatis dari "
            f"{result.media_info.original_width}×{result.media_info.original_height} "
            f"menjadi {result.media_info.width}×{result.media_info.height} "
            "sebelum inference."
        )
    if _requires_manual_review(bundle):
        st.warning(
            "Perlu verifikasi manual — sebagian label kelas atau track tidak "
            "cukup stabil untuk ditafsirkan sebagai jumlah absolut."
        )
    if bundle.annotated_mime.startswith("image/"):
        st.image(bundle.annotated_media, caption="Preview dengan bounding box")
    else:
        st.video(bundle.annotated_media, format="video/mp4")

    first, second, third = st.columns(3)
    count_label = (
        "Track terkonfirmasi—indikatif"
        if result.media_type.value == "video" and _requires_manual_review(bundle)
        else "Objek unik"
    )
    first.metric(count_label, result.unique_object_count)
    second.metric(
        "Mode", "Gambar" if result.media_type.value == "image" else "Video"
    )
    third.metric("Model", result.model_version or "tidak tersedia")
    st.caption(
        f"Analysis ID `{result.analysis_id}` · schema `{result.schema_version}` · "
        f"snapshot `{result.dataset_snapshot_id}`"
    )

    _render_composition(bundle)
    _render_priority(bundle)
    _render_recommendations(bundle)

    st.subheader("Quality flags dan keterbatasan")
    for flag in result.quality_flags:
        st.warning(f"{flag.code.value}: {flag.message}")
        if flag.code in {
            QualityFlagCode.CLASSIFICATION_UNCERTAIN,
            QualityFlagCode.TRACKING_UNSTABLE,
        }:
            with st.expander(f"Lihat evidence {flag.code.value}"):
                st.json(dict(flag.evidence))
    for limitation in result.limitations:
        st.markdown(f"- {limitation}")

    st.subheader("Export")
    json_column, csv_column = st.columns(2)
    json_column.download_button(
        "Download JSON lengkap",
        bundle.json_bytes,
        file_name=f"{result.analysis_id}.json",
        mime="application/json",
        use_container_width=True,
    )
    csv_column.download_button(
        "Download CSV ringkas",
        bundle.csv_bytes,
        file_name=f"{result.analysis_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _requires_manual_review(bundle: AnalysisBundle) -> bool:
    review_codes = {
        QualityFlagCode.CLASSIFICATION_UNCERTAIN,
        QualityFlagCode.TRACKING_UNSTABLE,
    }
    return any(flag.code in review_codes for flag in bundle.result.quality_flags)


def render_app(config_dir: Path | str = "configs") -> None:
    config = load_project_config(config_dir)
    st.set_page_config(page_title="RiverSort AI", page_icon="🌊", layout="wide")
    st.title("RiverSort AI")
    st.write(
        "Prototype pendukung keputusan untuk menganalisis sampah plastik "
        "mengambang dari satu gambar atau video pendek."
    )
    st.warning(
        "Estimasi komposisi visual berdasarkan jumlah objek unik dan proporsi area deteksi. "
        "Hasil bukan estimasi berat, massa, tonase, atau volume fisik dan tetap "
        "memerlukan verifikasi manusia."
    )
    _render_capability(config)

    try:
        services = _load_services(str(config_dir))
        runtime_error = None
    except AnalysisError as error:
        services = None
        runtime_error = error
        st.error(f"{error.code.value}: {error.public_message}")
        st.info(error.recovery_action)
    except Exception:
        services = None
        runtime_error = True
        LOGGER.error(
            "runtime initialization failed",
            extra={"stage": "startup", "error_code": "MODEL_NOT_READY"},
        )
        st.error("MODEL_NOT_READY: runtime lokal gagal dimulai.")
        st.info("Periksa environment, config, dan model artifact lalu muat ulang.")

    mode = st.radio("Jenis media", ("Gambar", "Video"), horizontal=True)
    allowed = ["jpg", "jpeg", "png"] if mode == "Gambar" else ["mp4"]
    uploaded = st.file_uploader(
        "Unggah satu media",
        type=allowed,
        help=(
            "JPEG/PNG maksimal 20 MB. Gambar 25–80 MP diperkecil otomatis ke "
            "maksimal 25 MP."
            if mode == "Gambar"
            else "MP4 H.264 maksimal 200 MB dan 60 detik. Video di atas "
            "1920×1080 hingga 8K diperkecil otomatis ke dalam batas 1080p."
        ),
    )
    uploaded_content = uploaded.getvalue() if uploaded is not None else None
    current_input_key = (
        f"{mode}:{hashlib.sha256(uploaded_content).hexdigest()}"
        if uploaded_content is not None
        else None
    )
    previous_input_key = st.session_state.get("analysis_input_key")
    if previous_input_key is not None and previous_input_key != current_input_key:
        _clear_result()
    if uploaded is not None:
        st.caption(
            f"File dipilih: `{Path(uploaded.name).name}` · "
            f"{len(uploaded_content or b'') / (1024 * 1024):.2f} MB · "
            f"batas mode {mode.lower()} akan diverifikasi sebelum inference."
        )

    with st.expander("Metadata observasi (opsional)"):
        location = st.text_input("Lokasi")
        observed_at = st.text_input("Waktu observasi", placeholder="2026-07-23 09:00 WIB")
        weather = st.selectbox(
            "Cuaca", ("", "cerah", "berawan", "hujan ringan", "setelah hujan")
        )
        notes = st.text_area("Catatan", max_chars=500)

    in_progress = bool(st.session_state.get("analysis_in_progress", False))
    analyze = st.button(
        "Mulai analisis",
        type="primary",
        disabled=uploaded is None or services is None or in_progress,
        use_container_width=True,
    )
    if (
        analyze
        and uploaded is not None
        and uploaded_content is not None
        and services is not None
    ):
        _clear_result()
        st.session_state["analysis_in_progress"] = True
        metadata = {
            "location": location,
            "observed_at": observed_at,
            "weather": weather,
            "notes": notes,
        }
        media = UploadedMedia(uploaded.name, uploaded_content, uploaded.type)
        try:
            with st.status("Analisis sedang berjalan", expanded=True) as status:
                use_case = services.image if mode == "Gambar" else services.video
                bundle = use_case.execute(
                    media, metadata, progress=_progress_writer(status)
                )
                status.update(label="Analisis selesai", state="complete")
            st.session_state["analysis_bundle"] = bundle
            st.session_state["analysis_mode"] = mode
            st.session_state["analysis_input_key"] = current_input_key
        except AnalysisError as error:
            st.error(f"{error.code.value}: {error.public_message}")
            st.info(error.recovery_action)
        except Exception:
            LOGGER.error(
                "analysis failed",
                extra={"stage": "analysis", "error_code": "INFERENCE_FAILED"},
            )
            st.error("INFERENCE_FAILED: analisis tidak dapat diselesaikan.")
            st.info("Coba media lain atau muat ulang aplikasi.")
        finally:
            st.session_state["analysis_in_progress"] = False

    bundle = st.session_state.get("analysis_bundle")
    if bundle is not None:
        _render_result(bundle)
        if st.button("Mulai analisis baru", use_container_width=True):
            _clear_result()
            st.rerun()

    if runtime_error:
        st.caption("Input tetap dapat dipilih, tetapi analisis dinonaktifkan sampai runtime siap.")


def render_smoke_page(config_dir: Path | str = "configs") -> None:
    """Backward-compatible entrypoint used by earlier smoke tests."""

    render_app(config_dir)
