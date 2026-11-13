
from datetime import datetime
from pathlib import Path
import base64
import io

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import torch

from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


# ==================================================
# KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="PlantVision | Klasifikasi Daun",
    page_icon="🌿",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "plantvision_leaf_classifier.keras"
CLASS_PATH = APP_DIR / "class_names.txt"

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 20

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Ambang awal dibuat lebih longgar agar daun lebih mudah lolos.
# Nilai ini perlu dievaluasi menggunakan foto pengujian.
LEAF_THRESHOLD = 0.15
MIN_SCORE_GAP = 0.00

DEFAULT_CLASSES = ["Sehat", "Menguning", "Rusak"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# MEMUAT MODEL KLASIFIKASI DAUN
# ==================================================

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


def load_classes():
    if CLASS_PATH.exists():
        classes = [
            line.strip()
            for line in CLASS_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(classes) == 3:
            return classes

    return DEFAULT_CLASSES


# ==================================================
# MEMUAT MODEL CLIP
# ==================================================

@st.cache_resource
def load_clip_model(model_name):
    processor = CLIPProcessor.from_pretrained(model_name)

    clip_model = CLIPModel.from_pretrained(model_name)
    clip_model.to(DEVICE)
    clip_model.eval()

    return clip_model, processor


# ==================================================
# PEMERIKSAAN AWAL FOTO DENGAN CLIP
# ==================================================

def check_leaf_clip(image, clip_model, processor):
    """
    Memeriksa kemiripan foto dengan berbagai deskripsi daun
    dan non-daun menggunakan CLIP lokal.

    Tidak melakukan training ulang dan tidak menggunakan API.
    """

    leaf_descriptions = [
        "a close-up photo of a single plant leaf",
        "a photograph of green leaves on a plant",
        "a photograph of a healthy green leaf",
        "a photograph of a yellow leaf",
        "a photograph of a brown dried leaf",
        "a photograph of a damaged or diseased leaf",
        "a photograph of a leaf with brown spots",
        "a photograph of a leaf with yellow spots",
        "a photograph of a wilted plant leaf",
        "a photograph of a leaf on a tree",
        "a photograph of a leaf held in someone's hand",
        "a photograph of several leaves with a natural background",
        "a photograph of a plant with visible leaves",
        "a photograph of a small leaf among other plants",
        "a photograph of a leaf against a plain background",
        "a photograph of a leaf on the ground",
        "a photograph of a dry fallen leaf",
        "a photograph of the surface and veins of a leaf",
    ]

    non_leaf_descriptions = [
        "a photograph of a person",
        "a photograph of a human face",
        "a photograph of a dog or cat",
        "a photograph of an animal",
        "a photograph of food",
        "a photograph of a building",
        "a photograph of a vehicle",
        "a photograph of an electronic device",
        "a photograph of clothing",
        "a photograph of a shoe or bag",
        "a photograph of a table or chair",
        "a photograph of a landscape without a visible leaf",
        "a photograph of a flower without visible leaves",
        "a photograph of an everyday object",
    ]

    descriptions = leaf_descriptions + non_leaf_descriptions

    inputs = processor(
        text=descriptions,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = clip_model(**inputs)

        # Skor kemiripan gambar dengan deskripsi.
        logits = outputs.logits_per_image[0]
        probabilities = torch.softmax(logits, dim=0)

    scores = probabilities.detach().cpu().numpy()

    leaf_count = len(leaf_descriptions)

    # Skor kategori daun dan non-daun.
    leaf_scores = scores[:leaf_count]
    non_leaf_scores = scores[leaf_count:]

    best_leaf_index = int(np.argmax(leaf_scores))
    best_non_leaf_index = int(np.argmax(non_leaf_scores))

    best_leaf_score = float(leaf_scores[best_leaf_index])
    best_non_leaf_score = float(non_leaf_scores[best_non_leaf_index])

    # Jumlah probabilitas untuk seluruh variasi deskripsi daun.
    # Ini digunakan sebagai informasi tambahan, bukan probabilitas
    # terkalibrasi bahwa gambar benar-benar berisi daun.
    leaf_group_score = float(np.sum(leaf_scores))

    # Deskripsi dengan skor individual tertinggi.
    best_overall_index = int(np.argmax(scores))
    best_description = descriptions[best_overall_index]

    score_gap = best_leaf_score - best_non_leaf_score

    # Filter dibuat lebih longgar daripada versi sebelumnya.
    is_likely_leaf = (
        best_leaf_score >= LEAF_THRESHOLD
        and score_gap >= MIN_SCORE_GAP
        and best_overall_index < leaf_count
    )

    details = {
        "Skor deskripsi daun tertinggi": best_leaf_score,
        "Skor deskripsi non-daun tertinggi": best_non_leaf_score,
        "Selisih skor tertinggi": float(score_gap),
        "Total skor kelompok daun": leaf_group_score,
        "Deskripsi paling cocok": best_description,
    }

    return is_likely_leaf, details


# ==================================================
# MEMBACA DAN MENAMPILKAN GAMBAR
# ==================================================

def read_image(data):
    with Image.open(io.BytesIO(data)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.load()

    return image


def show_preview(image):
    st.subheader("Preview foto")

    st.caption(
        f"Dimensi: {image.width} × {image.height} piksel"
    )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    st.markdown(
        f"""
        <div style="
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 12px;
            text-align: center;
        ">
            <img
                src="data:image/jpeg;base64,{encoded}"
                alt="Preview foto"
                style="
                    max-width: 100%;
                    max-height: 450px;
                    width: auto;
                    height: auto;
                    object-fit: contain;
                    border-radius: 8px;
                "
            />
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# PREDIKSI KONDISI DAUN DENGAN MOBILENETV2
# ==================================================

def predict_image(model, image, classes):
    resized_image = image.convert("RGB").resize(IMAGE_SIZE)

    array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    batch = np.expand_dims(array, axis=0)

    # Preprocessing sudah ada di dalam model dari Colab.
    scores = np.asarray(
        model.predict(batch, verbose=0)
    ).reshape(-1)

    if len(scores) != len(classes):
        raise ValueError(
            "Jumlah output model tidak sesuai dengan daftar kelas."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Skor prediksi tidak valid.")

    # Menangani output softmax atau skor yang belum ternormalisasi.
    if (
        np.any(scores < 0)
        or not np.isclose(scores.sum(), 1.0, atol=0.05)
    ):
        exp_scores = np.exp(scores - np.max(scores))
        scores = exp_scores / exp_scores.sum()
    else:
        scores = scores / scores.sum()

    index = int(np.argmax(scores))

    details = {
        name: float(score)
        for name, score in zip(classes, scores)
    }

    return classes[index], float(scores[index]), details


# ==================================================
# HEADER APLIKASI
# ==================================================

st.markdown(
    """
    <div style="
        padding: 24px;
        border-radius: 18px;
        background: linear-gradient(120deg, #e9f7ed, #f5fbf6);
        border: 1px solid #d3e8d8;
    ">
        <p style="color: #38734b; font-weight: bold;">
            PLANTVISION · COMPUTER VISION
        </p>
        <h1 style="color: #173f29;">
            Klasifikasi Kondisi Daun 🌿
        </h1>
        <p style="color: #41634c;">
            Upload foto daun untuk memprediksi kondisi visualnya.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

classes = load_classes()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("Tentang model")

    st.write("**Klasifikasi kondisi:** MobileNetV2")
    st.write("**Filter foto:** CLIP lokal")
    st.write("**Input:** RGB 224 × 224")
    st.write("**Kelas:** " + ", ".join(classes))

    st.caption(
        "Confidence klasifikasi adalah skor model, "
        "bukan diagnosis pasti."
    )

    st.divider()

    st.write("**Pemeriksaan awal foto**")

    st.caption(
        "CLIP membandingkan foto dengan beberapa deskripsi "
        "daun dan non-daun sebelum klasifikasi dilakukan."
    )


# ==================================================
# VALIDASI MODEL
# ==================================================

if not MODEL_PATH.exists():
    st.error(f"File model tidak ditemukan: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))

except Exception as error:
    st.error("Model klasifikasi gagal dimuat.")
    st.code(f"{type(error).__name__}: {error}")
    st.stop()

if len(classes) != 3:
    st.error("class_names.txt harus berisi tiga kelas.")
    st.stop()

if model.output_shape[-1] != len(classes):
    st.error(
        f"Model memiliki {model.output_shape[-1]} output, "
        f"tetapi tersedia {len(classes)} label. "
        "Pastikan menggunakan model tiga kelas."
    )
    st.stop()

# Model CLIP dimuat satu kali dan disimpan dalam cache.
try:
    with st.spinner(
        "Menyiapkan model CLIP. "
        "Pemuatan pertama dapat memerlukan waktu lebih lama..."
    ):
        clip_model, clip_processor = load_clip_model(
            CLIP_MODEL_NAME
        )

except Exception as error:
    st.error(
        "Model CLIP gagal dimuat. Pastikan dependensi terpasang "
        "dan koneksi internet tersedia untuk unduhan pertama."
    )
    st.code(f"{type(error).__name__}: {error}")
    st.stop()


# ==================================================
# RIWAYAT SESI
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []


tab_check, tab_history, tab_info = st.tabs(
    [
        "🔎 Periksa daun",
        "🕘 Riwayat",
        "ℹ️ Panduan",
    ]
)


# ==================================================
# TAB PEMERIKSAAN FOTO
# ==================================================

with tab_check:
    st.subheader("Masukkan foto daun")

    camera_col, upload_col = st.columns(2)

    with camera_col:
        camera_file = st.camera_input(
            "Ambil foto menggunakan kamera",
            key="camera_leaf",
        )

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload foto dari perangkat",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="upload_leaf",
        )

    source = (
        uploaded_file
        if uploaded_file is not None
        else camera_file
    )

    if uploaded_file is not None and camera_file is not None:
        choice = st.radio(
            "Pilih foto yang digunakan",
            ["Foto upload", "Foto kamera"],
            horizontal=True,
        )

        source = (
            uploaded_file
            if choice == "Foto upload"
            else camera_file
        )

    if source is not None:
        image_bytes = source.getvalue()
        size_mb = len(image_bytes) / (1024 * 1024)

        file_name = getattr(
            source,
            "name",
            "Foto kamera",
        )

        st.write(f"Nama file: {file_name}")
        st.write(f"Ukuran file: {size_mb:.2f} MB")

        if size_mb > MAX_UPLOAD_MB:
            st.error(
                f"Ukuran gambar melebihi {MAX_UPLOAD_MB} MB."
            )

        else:
            try:
                image = read_image(image_bytes)

                show_preview(image)

                if st.button(
                    "🌱 Prediksi kondisi daun",
                    type="primary",
                    use_container_width=True,
                ):
                    # Tahap 1: pemeriksaan foto menggunakan CLIP.
                    with st.spinner(
                        "Memeriksa foto menggunakan CLIP..."
                    ):
                        is_likely_leaf, clip_details = check_leaf_clip(
                            image,
                            clip_model,
                            clip_processor,
                        )

                    # Tampilkan alasan hasil pemeriksaan.
                    with st.expander(
                        "Detail pemeriksaan awal",
                        expanded=not is_likely_leaf,
                    ):
                        st.write(
                            "Skor deskripsi daun tertinggi: "
                            f"{clip_details['Skor deskripsi daun tertinggi']:.1%}"
                        )

                        st.write(
                            "Skor deskripsi non-daun tertinggi: "
                            f"{clip_details['Skor deskripsi non-daun tertinggi']:.1%}"
                        )

                        st.write(
                            "Selisih skor tertinggi: "
                            f"{clip_details['Selisih skor tertinggi']:.1%}"
                        )

                        st.write(
                            "Total skor kelompok daun: "
                            f"{clip_details['Total skor kelompok daun']:.1%}"
                        )

                        st.write(
                            "Deskripsi paling cocok: "
                            f"{clip_details['Deskripsi paling cocok']}"
                        )

                    # Tahap 2: hentikan foto yang tidak lolos.
                    if not is_likely_leaf:
                        st.warning(
                            "⚠️ Foto belum lolos pemeriksaan awal. "
                            "Prediksi dibatalkan. Coba foto daun "
                            "yang lebih jelas dan menjadi fokus gambar."
                        )

                    # Tahap 3: klasifikasi kondisi daun.
                    else:
                        st.success(
                            "Foto lolos pemeriksaan awal. "
                            "Melanjutkan klasifikasi kondisi daun."
                        )

                        with st.spinner(
                            "Menganalisis kondisi daun..."
                        ):
                            label, confidence, scores = predict_image(
                                model,
                                image,
                                classes,
                            )

                        record = {
                            "Waktu": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "Sumber": file_name,
                            "Prediksi": label,
                            "Confidence": confidence,
                            **{
                                f"Skor {name}": score
                                for name, score in scores.items()
                            },
                        }

                        st.session_state.history.insert(
                            0,
                            record,
                        )

                        st.divider()

                        if label == "Sehat":
                            st.success(
                                f"Hasil prediksi: {label}"
                            )

                        elif label == "Menguning":
                            st.warning(
                                f"Hasil prediksi: {label}"
                            )

                        else:
                            st.error(
                                f"Hasil prediksi: {label}"
                            )

                        st.metric(
                            "Confidence model",
                            f"{confidence:.1%}",
                        )

                        st.progress(float(confidence))

                        st.subheader("Skor setiap kelas")

                        score_cols = st.columns(3)

                        for col, name in zip(
                            score_cols,
                            classes,
                        ):
                            col.metric(
                                name,
                                f"{scores[name]:.1%}",
                            )

                            col.progress(float(scores[name]))

                        if confidence < 0.60:
                            st.warning(
                                "Skor tertinggi rendah. "
                                "Hasil prediksi belum pasti."
                            )

                        st.info(
                            "Hasil merupakan klasifikasi visual, "
                            "bukan diagnosis pasti penyakit tanaman."
                        )

            except Exception as error:
                st.error(
                    "Terjadi kesalahan saat memproses gambar."
                )
                st.code(
                    f"{type(error).__name__}: {error}"
                )


# ==================================================
# TAB RIWAYAT
# ==================================================

with tab_history:
    st.subheader("Riwayat pemeriksaan")

    history = st.session_state.history

    if not history:
        st.info("Belum ada pemeriksaan.")

    else:
        df = pd.DataFrame(history)

        metric_cols = st.columns(4)

        metric_cols[0].metric(
            "Total",
            len(df),
        )

        for index, name in enumerate(classes, start=1):
            count = int(
                (df["Prediksi"] == name).sum()
            )

            metric_cols[index].metric(
                name,
                count,
            )

        display_df = df.copy()

        for column in display_df.columns:
            if (
                column == "Confidence"
                or column.startswith("Skor ")
            ):
                display_df[column] = display_df[column].map(
                    lambda value: f"{value:.1%}"
                )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Unduh riwayat CSV",
            data=df.to_csv(
                index=False,
            ).encode("utf-8-sig"),
            file_name="riwayat_klasifikasi_daun.csv",
            mime="text/csv",
        )

        if st.button("Hapus riwayat"):
            st.session_state.history = []
            st.rerun()


# ==================================================
# TAB PANDUAN
# ==================================================

with tab_info:
    st.subheader("Cara menggunakan")

    st.markdown(
        """
        1. Upload foto atau ambil foto menggunakan kamera.
        2. Pastikan preview foto muncul.
        3. Tekan tombol prediksi.
        4. CLIP memeriksa kesesuaian foto dengan kategori daun.
        5. Jika lolos, MobileNetV2 memprediksi kondisi daun.
        6. Buka tab Riwayat untuk melihat hasil sebelumnya.
        7. Unduh riwayat dalam format CSV jika diperlukan.

        **Catatan:**

        CLIP merupakan filter awal berbasis kemiripan gambar
        dan teks. Foto daun masih mungkin ditolak, dan foto
        non-daun masih mungkin lolos. Ambang pemeriksaan perlu
        dievaluasi dengan contoh foto yang sesuai.

        Riwayat hanya disimpan selama sesi aplikasi aktif.
        Hasil klasifikasi bukan diagnosis pasti penyakit tanaman.
        """
    )
