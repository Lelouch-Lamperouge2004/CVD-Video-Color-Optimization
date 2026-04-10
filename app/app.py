import sys
import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from inference.video_pipeline import process_video


# -----------------------------
# Paths (LOCAL)
# -----------------------------
CKPT_PATH = r"D:\CVD_GAN\checkpoints\enhance_gan_conditional\latest.pth"
PLATE_DIR = Path(r"D:\CVD_GAN\app\plates")
OUTPUT_DIR = Path(r"D:\CVD_GAN\outputs\videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_LIST = ["protan", "deutan", "tritan"]


# -----------------------------
# Synthetic Plates Definition
# Must match generator numbers + filenames
# -----------------------------
RG_NUMS = ["12", "8", "6", "57", "45", "73"]
BY_NUMS = ["29", "5", "3", "64", "91", "27"]

PLATES = []
for i, n in enumerate(RG_NUMS, 1):
    PLATES.append({"file": f"plate_rg_{i}.png", "correct": n, "axis": "rg"})
for i, n in enumerate(BY_NUMS, 1):
    PLATES.append({"file": f"plate_by_{i}.png", "correct": n, "axis": "by"})


def classify_from_scores(rg_fail: int, by_fail: int):
    """
    Screening-based classification (NOT medical diagnosis).
    - If BY fails dominate -> tritan
    - If RG fails dominate -> red-green deficiency (protan/deutan)
    - else -> normal / uncertain
    """
    rg_thresh = 2
    by_thresh = 2

    if by_fail >= by_thresh and by_fail > rg_fail:
        return "tritan"

    if rg_fail >= rg_thresh and rg_fail > by_fail:
        return "red-green"

    if rg_fail < rg_thresh and by_fail < by_thresh:
        return "normal"

    return "uncertain"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="CVD Video Pre-Compensation", layout="centered")
st.title("CVD Video Pre-Compensation (GAN)")

st.info(
    "This screening is a lightweight, synthetic Ishihara-style test for model selection. "
    "It is NOT a medical diagnosis."
)

# ---------- Screening ----------
st.header("Step 1: Screening Test (12 Plates)")
st.write("Enter the number you see for each plate. If unclear, type 'none'.")

rg_fail = 0
by_fail = 0
answered = 0

with st.expander("Take the test", expanded=True):
    for idx, plate in enumerate(PLATES):
        img_path = PLATE_DIR / plate["file"]
        if not img_path.exists():
            st.error(f"Missing plate image: {img_path}")
            st.stop()

        st.image(
            Image.open(img_path),
            caption=f"Plate {idx+1} ({plate['axis'].upper()})",
            use_container_width=True,
        )

        ans = st.text_input(f"Answer for Plate {idx+1}", key=f"plate_{idx}").strip()

        if ans != "":
            answered += 1
            ans_norm = ans.lower().replace(" ", "")
            correct_norm = plate["correct"].lower()

            ok = (ans_norm == correct_norm)
            if not ok:
                if plate["axis"] == "rg":
                    rg_fail += 1
                else:
                    by_fail += 1

st.write(f"RG fails: **{rg_fail} / 6**   |   BY fails: **{by_fail} / 6**")
suggestion = classify_from_scores(rg_fail, by_fail)

if st.button("Get Suggested Type"):
    st.session_state["suggestion"] = suggestion
    if suggestion == "tritan":
        st.warning("Suggested type: **TRITAN** (blue-yellow axis)")
    elif suggestion == "red-green":
        st.warning("Suggested type: **RED-GREEN deficiency** (protan/deutan)")
    elif suggestion == "normal":
        st.success("No strong deficiency detected (screening result).")
    else:
        st.warning("Uncertain result. Please select manually.")


# ---------- Choose type ----------
st.header("Step 2: Choose Type & Upload Video")

suggestion = st.session_state.get("suggestion", "uncertain")

if suggestion == "tritan":
    default_type = "tritan"
elif suggestion == "red-green":
    default_type = "deutan"
else:
    default_type = "protan"

cvd_type = st.selectbox(
    "CVD type to apply (you can override screening)",
    TYPE_LIST,
    index=TYPE_LIST.index(default_type),
)

alpha = st.slider("Enhancement Strength (alpha)", 0.5, 1.0, 0.9, 0.05)
smooth = st.slider("Anti-flicker smoothing", 0.0, 0.6, 0.35, 0.05)

st.subheader("Optional: Edge Sharpness (no retraining)")
st.caption("If output looks soft, increase sharpen. If it looks noisy, reduce sharpen or increase threshold.")
sharpen = st.slider("Sharpen amount", 0.0, 1.0, 0.5, 0.05)
sharpen_radius = st.slider("Sharpen radius", 0.5, 2.5, 1.0, 0.1)
sharpen_thresh = st.slider("Sharpen threshold", 0, 10, 3, 1)

uploaded = st.file_uploader(
    "Upload MP4 video (recommended ≤ 30 sec for demo)",
    type=["mp4"],
)

# ---------- Run ----------
if uploaded and st.button("Process Video"):
    if not Path(CKPT_PATH).exists():
        st.error(f"Missing checkpoint: {CKPT_PATH}")
        st.stop()

    st.info("Processing... (real progress)")

    progress_bar = st.progress(0)
    status_text = st.empty()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded.read())
        in_path = tmp_in.name

    out_path = str(OUTPUT_DIR / f"streamlit_out_{cvd_type}.mp4")

    def on_progress(done, total):
        # total may be None if OpenCV can't read frame count
        if total is None:
            status_text.text(f"Processed {done} frames...")
            progress_bar.progress(min(99, done % 100))
        else:
            pct = int((done / total) * 100)
            progress_bar.progress(min(100, pct))
            status_text.text(f"Processing {done}/{total} frames ({pct}%)")

    try:
        process_video(
            in_video=in_path,
            out_video=out_path,
            ckpt_path=CKPT_PATH,
            cvd_type=cvd_type,
            model_size=256,
            alpha=float(alpha),
            smooth=float(smooth),
            lab_boost=0.25,
            sharpen=float(sharpen),
            sharpen_radius=float(sharpen_radius),
            sharpen_thresh=int(sharpen_thresh),
            progress_callback=on_progress,
        )
        progress_bar.progress(100)
        status_text.text("Processing complete (100%)")
    finally:
        try:
            os.remove(in_path)
        except:
            pass

    st.success("Done.")

    with open(out_path, "rb") as f:
        st.download_button(
            "Download output video",
            data=f,
            file_name=f"output_{cvd_type}.mp4",
            mime="video/mp4",
        )