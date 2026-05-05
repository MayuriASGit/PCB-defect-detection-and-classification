import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from PIL import Image
import io
import os
import sys
import time
import zipfile
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocessing import preprocess_images, subtract_images, apply_threshold
from pipeline.contour import extract_contours, extract_rois, draw_contours, draw_difference_overlay
from pipeline.model import DEFECT_CLASSES, DEFECT_DESCRIPTIONS, get_model
from pipeline.inference import run_full_pipeline, save_outputs, InferenceResult

st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "history" not in st.session_state:
    st.session_state.history = []


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return Image.open(buf).copy()


def make_download_zip(result: InferenceResult, annotated_pil, diff_pil, mask_pil, comparison_pil, df) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in [
            ("annotated_result.png", annotated_pil),
            ("difference_map.png", diff_pil),
            ("binary_mask.png", mask_pil),
            ("comparison.png", comparison_pil),
        ]:
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(name, img_buf.getvalue())

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        zf.writestr("defect_log.csv", csv_buf.getvalue())

        summary = {
            "defect_count": result.defect_count,
            "processing_time_ms": round(result.processing_time_ms, 1),
            "timestamp": result.timestamp,
            "defects": [
                {"id": i + 1, "class": l, "confidence": round(c, 4)}
                for i, (l, c) in enumerate(zip(result.labels, result.confidences))
            ],
        }
        zf.writestr("summary.json", json.dumps(summary, indent=2))

    buf.seek(0)
    return buf.read()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("Image Processing")
    threshold_method = st.selectbox(
        "Threshold Method",
        ["otsu", "adaptive", "manual"],
        help="Otsu's method automatically finds the optimal threshold."
    )
    manual_thresh = 30
    if threshold_method == "manual":
        manual_thresh = st.slider("Manual Threshold", 5, 200, 30)

    st.subheader("Defect Filtering")
    min_area = st.slider("Min Defect Area (px²)", 10, 500, 50,
                         help="Ignore contours smaller than this area.")
    max_area = st.slider("Max Defect Area (px²)", 1000, 100000, 50000,
                         help="Ignore contours larger than this area.")

    st.markdown("---")
    st.subheader("Model")
    st.info(
        "Using **EfficientNet-B0** architecture.\n\n"
        "Classification is performed with heuristic simulation (upload a trained `.pth` checkpoint to use the actual model)."
    )
    uploaded_model = st.file_uploader("Upload Model Checkpoint (.pth)", type=["pth"])

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        "**PCB Defect Detection System**\n\n"
        "Detects and classifies 6 defect types:\n" +
        "\n".join(f"- `{c}`" for c in DEFECT_CLASSES)
    )


# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 AI PCB Defect Detection & Classification")
st.markdown(
    "Upload a **reference (template)** PCB image and a **test** PCB image. "
    "The system will detect and classify defects using image subtraction and EfficientNet-B0."
)
st.markdown("---")

# ─── Upload Section ────────────────────────────────────────────────────────────
col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("📁 Template Image (Defect-Free)")
    template_file = st.file_uploader("Upload template PCB", type=["png", "jpg", "jpeg", "bmp"], key="template")
    if template_file:
        template_pil = Image.open(template_file)
        st.image(template_pil, caption="Template (Reference)", use_container_width=True)

with col_up2:
    st.subheader("📁 Test Image (With Defects)")
    test_file = st.file_uploader("Upload test PCB", type=["png", "jpg", "jpeg", "bmp"], key="test")
    if test_file:
        test_pil = Image.open(test_file)
        st.image(test_pil, caption="Test Image", use_container_width=True)

# ─── Demo Images ───────────────────────────────────────────────────────────────
st.markdown("---")
use_demo = st.checkbox("Use built-in demo images (synthetic PCB)", value=False)
if use_demo:
    st.info(
        "Demo mode: synthetic PCB images will be generated for demonstration. "
        "Upload real PCB images for accurate results."
    )


def generate_demo_pcb(with_defects: bool = False) -> Image.Image:
    """Generate a synthetic PCB-like image for demo."""
    np.random.seed(42 if not with_defects else 7)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 40
    img[:, :] = [34, 85, 34]

    for _ in range(30):
        x1 = np.random.randint(0, 480)
        y1 = np.random.randint(0, 480)
        x2 = x1 + np.random.randint(20, 80)
        y2 = y1 + np.random.randint(3, 8)
        cv2.rectangle(img, (x1, y1), (min(x2, 511), min(y2, 511)), (180, 140, 60), -1)

    for _ in range(20):
        x1 = np.random.randint(0, 480)
        y1 = np.random.randint(0, 480)
        x2 = x1 + np.random.randint(3, 8)
        y2 = y1 + np.random.randint(20, 80)
        cv2.rectangle(img, (x1, y1), (min(x2, 511), min(y2, 511)), (180, 140, 60), -1)

    for _ in range(15):
        cx = np.random.randint(20, 490)
        cy = np.random.randint(20, 490)
        r = np.random.randint(5, 15)
        cv2.circle(img, (cx, cy), r, (200, 200, 200), -1)
        cv2.circle(img, (cx, cy), r - 2, (40, 40, 40), -1)

    if with_defects:
        defect_positions = [(80, 80), (200, 150), (350, 300), (420, 100), (130, 380)]
        for dx, dy in defect_positions:
            choice = np.random.randint(0, 4)
            if choice == 0:
                cv2.circle(img, (dx, dy), 12, (34, 85, 34), -1)
                cv2.circle(img, (dx, dy), 4, (34, 85, 34), -1)
            elif choice == 1:
                pts = np.array([[dx, dy], [dx + 25, dy + 5], [dx + 20, dy + 20]], np.int32)
                cv2.fillPoly(img, [pts], (180, 140, 60))
            elif choice == 2:
                cv2.rectangle(img, (dx, dy), (dx + 40, dy + 6), (34, 85, 34), -1)
            else:
                cv2.rectangle(img, (dx - 5, dy - 5), (dx + 35, dy + 35), (180, 140, 60), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if use_demo:
    demo_template = Image.fromarray(generate_demo_pcb(with_defects=False))
    demo_test = Image.fromarray(generate_demo_pcb(with_defects=True))
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.image(demo_template, caption="Demo Template", use_container_width=True)
    with col_d2:
        st.image(demo_test, caption="Demo Test (with synthetic defects)", use_container_width=True)

# ─── Run Button ────────────────────────────────────────────────────────────────
st.markdown("---")
run_button = st.button("🚀 Run Defect Detection", type="primary", use_container_width=True)

if run_button:
    if not use_demo and (template_file is None or test_file is None):
        st.error("Please upload both template and test images, or enable demo mode.")
    else:
        template_pil_input = demo_template if use_demo else template_pil
        test_pil_input = demo_test if use_demo else test_pil

        model_instance = None
        if uploaded_model:
            try:
                import torch
                model_instance = get_model(num_classes=len(DEFECT_CLASSES))
                checkpoint = torch.load(io.BytesIO(uploaded_model.read()), map_location="cpu")
                state = checkpoint.get("model_state_dict", checkpoint)
                model_instance.load_state_dict(state)
                model_instance.eval()
                st.success("Custom model checkpoint loaded.")
                use_sim = False
            except Exception as e:
                st.warning(f"Could not load model checkpoint: {e}. Falling back to simulation.")
                model_instance = None
                use_sim = True
        else:
            use_sim = True

        with st.spinner("Running detection pipeline..."):
            settings = {
                "threshold_method": threshold_method,
                "manual_threshold": manual_thresh,
                "min_defect_area": min_area,
                "max_defect_area": max_area,
                "model": model_instance,
                "use_simulation": use_sim,
                "device": "cpu",
            }
            try:
                result = run_full_pipeline(template_pil_input, test_pil_input, settings)
                st.session_state.result = result
                st.session_state.run_count += 1
                st.session_state.history.append({
                    "run": st.session_state.run_count,
                    "defects": result.defect_count,
                    "time_ms": round(result.processing_time_ms, 1),
                    "timestamp": result.timestamp,
                })
                st.success(f"✅ Detection complete in {result.processing_time_ms:.0f} ms")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

# ─── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    result: InferenceResult = st.session_state.result
    st.markdown("---")
    st.header("📊 Detection Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Defects Found", result.defect_count)
    m2.metric("Processing Time", f"{result.processing_time_ms:.0f} ms")
    if result.confidences:
        m3.metric("Avg Confidence", f"{np.mean(result.confidences):.1%}")
        m4.metric("Top Defect", result.labels[0].replace("_", " ").title() if result.labels else "—")
    else:
        m3.metric("Avg Confidence", "—")
        m4.metric("Top Defect", "None")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖼️ Annotated Output",
        "🔍 Difference Analysis",
        "📋 Defect Table",
        "📈 Analytics",
        "🧠 Model Info",
    ])

    annotated_pil = cv2_to_pil(result.annotated_image)
    diff_pil = Image.fromarray(result.diff_image)
    mask_pil = Image.fromarray(result.binary_mask)
    comparison_pil = cv2_to_pil(result.comparison_image)

    with tab1:
        st.subheader("Annotated Test Image")
        st.image(annotated_pil, caption="Detected defects with bounding boxes and labels", use_container_width=True)

        if result.rois:
            st.subheader(f"Extracted ROIs ({len(result.rois)} defects)")
            roi_cols = st.columns(min(len(result.rois), 6))
            for i, (roi, label, conf) in enumerate(zip(result.rois[:6], result.labels[:6], result.confidences[:6])):
                with roi_cols[i]:
                    roi_pil = cv2_to_pil(roi.roi_image)
                    st.image(roi_pil, caption=f"{label.replace('_',' ').title()}\n{conf:.1%}", use_container_width=True)
        else:
            st.info("No defects detected. Try adjusting the threshold or area settings in the sidebar.")

    with tab2:
        st.subheader("Difference Map & Mask")
        col_diff1, col_diff2 = st.columns(2)
        with col_diff1:
            st.image(diff_pil, caption="Grayscale Difference Map", use_container_width=True)
        with col_diff2:
            st.image(mask_pil, caption="Binary Defect Mask (after thresholding)", use_container_width=True)

        st.subheader("Side-by-Side Comparison")
        st.image(comparison_pil, caption="Template | Test Image | Defect Overlay", use_container_width=True)

    with tab3:
        df = result.to_dataframe()
        st.subheader(f"Defect Log ({len(df)} entries)")
        if not df.empty:
            display_df = df[["defect_id", "class", "description", "confidence_pct", "area_px", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]].copy()
            display_df.columns = ["ID", "Class", "Description", "Confidence", "Area (px²)", "X", "Y", "Width", "Height"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download CSV Log",
                data=csv_bytes,
                file_name=f"defect_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No defects to display.")

    with tab4:
        if result.defect_count > 0:
            col_a1, col_a2 = st.columns(2)

            with col_a1:
                st.subheader("Defect Class Distribution")
                from collections import Counter
                counts = Counter(result.labels)
                fig, ax = plt.subplots(figsize=(6, 4))
                classes = [c.replace("_", " ").title() for c in counts.keys()]
                vals = list(counts.values())
                bars = ax.barh(classes, vals, color=plt.cm.Set2.colors[:len(classes)])
                ax.set_xlabel("Count")
                ax.set_title("Defect Distribution")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                            str(v), va="center")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_a2:
                st.subheader("Confidence Scores")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                x = range(len(result.labels))
                bar_labels = [f"#{i+1} {l.replace('_',' ')[:10]}" for i, l in enumerate(result.labels)]
                colors = plt.cm.RdYlGn([c for c in result.confidences])
                ax2.bar(x, [c * 100 for c in result.confidences], color=colors)
                ax2.set_xticks(list(x))
                ax2.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
                ax2.set_ylabel("Confidence (%)")
                ax2.set_ylim(0, 105)
                ax2.set_title("Per-Defect Confidence")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            st.subheader("Probability Heatmap (All Classes)")
            prob_matrix = np.array(result.all_probs)
            fig3, ax3 = plt.subplots(figsize=(10, max(3, len(result.labels) * 0.5 + 1)))
            col_labels = [c.replace("_", "\n") for c in DEFECT_CLASSES]
            row_labels = [f"Defect #{i+1}" for i in range(len(result.labels))]
            sns.heatmap(
                prob_matrix, ax=ax3,
                xticklabels=col_labels, yticklabels=row_labels,
                annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5
            )
            ax3.set_title("Classification Probability Matrix")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.subheader("Defect Size Distribution")
            fig4, ax4 = plt.subplots(figsize=(8, 3))
            areas = [roi.area for roi in result.rois]
            ax4.hist(areas, bins=min(10, len(areas)), color="steelblue", edgecolor="white")
            ax4.set_xlabel("Area (px²)")
            ax4.set_ylabel("Count")
            ax4.set_title("Defect Region Area Distribution")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

        else:
            st.info("No defects detected — no analytics to display.")

    with tab5:
        st.subheader("Model Architecture: EfficientNet-B0")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
**Architecture:** EfficientNet-B0 (transfer learning)

**Input size:** 128 × 128 × 3

**Output:** 6 defect classes via softmax

**Optimizer:** Adam

**Loss function:** Cross-Entropy

**Training augmentations:**
- Random horizontal/vertical flip
- Random rotation ±15°
- Color jitter
- Normalization (ImageNet stats)
""")
        with col_m2:
            st.markdown("""
**Defect Classes:**

| # | Class | Description |
|---|-------|-------------|
| 0 | missing_hole | Absent drill hole |
| 1 | mouse_bite | Semicircular edge bite |
| 2 | open_circuit | Broken conductor |
| 3 | short | Unwanted connection |
| 4 | spur | Extra copper protrusion |
| 5 | spurious_copper | Leftover copper |

**Training target:** ≥ 95% test accuracy

**Dataset:** DeepPCB
""")

        st.subheader("Training the Model")
        st.markdown("""
To train a model on the **DeepPCB dataset** and use it here:

```bash
# From the pcb-defect-detection directory
python train.py --data_dir /path/to/DeepPCB --epochs 50 --batch_size 32
```

This will save `model_checkpoint.pth` which you can upload in the sidebar.
""")
        with st.expander("View Training Script"):
            try:
                with open(os.path.join(os.path.dirname(__file__), "train.py")) as f:
                    st.code(f.read(), language="python")
            except FileNotFoundError:
                st.warning("train.py not found.")

    st.markdown("---")
    st.subheader("⬇️ Download All Results")
    df_for_zip = result.to_dataframe()
    zip_bytes = make_download_zip(result, annotated_pil, diff_pil, mask_pil, comparison_pil, df_for_zip)
    st.download_button(
        "📦 Download ZIP (images + CSV + JSON summary)",
        data=zip_bytes,
        file_name=f"pcb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
    )

    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.subheader("📜 Session History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
