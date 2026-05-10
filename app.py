import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Human Detection",
    page_icon="👤",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark background */
  .stApp {
    background-color: #0a0c10;
    color: #e8eaf0;
  }

  /* Hide default header/footer/menu */
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero title */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.2rem;
  }
  .hero-accent {
    color: #00e5ff;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
  }

  /* Stat card */
  .stat-card {
    background: #12151c;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00e5ff;
    line-height: 1;
  }
  .stat-label {
    font-size: 0.75rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    border: 1px dashed #1e2330 !important;
    border-radius: 12px !important;
    background: #12151c !important;
    padding: 1rem !important;
  }

  /* Buttons */
  .stButton > button {
    background: #00e5ff !important;
    color: #0a0c10 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover {
    opacity: 0.85 !important;
  }

  /* Progress bar */
  .stProgress > div > div {
    background-color: #00e5ff !important;
  }

  /* Divider */
  hr {
    border-color: #1e2330;
    margin: 1.5rem 0;
  }

  /* Info/success boxes */
  .stAlert {
    background: #12151c !important;
    border-color: #1e2330 !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
  }

  /* Select box */
  .stSelectbox > div > div {
    background: #12151c !important;
    border-color: #1e2330 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
  }

  /* Image caption */
  .stImage > div > div > div {
    color: #4b5563 !important;
    font-size: 0.78rem !important;
  }

  /* Section label */
  .section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.6rem;
  }

  /* Tag badge */
  .badge {
    display: inline-block;
    background: #0d2a30;
    color: #00e5ff;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-right: 0.4rem;
    border: 1px solid #00e5ff22;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 0.5rem;">
  <div class="hero-title">Human <span class="hero-accent">Detection</span></div>
  <div class="hero-sub">YOLOv8 · Real-time · Video Analysis</div>
</div>
<div style="margin-bottom: 1.8rem;">
  <span class="badge">YOLOv8n</span>
  <span class="badge">OpenCV</span>
  <span class="badge">Ultralytics</span>
</div>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ── Mode selector ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Detection Mode</div>', unsafe_allow_html=True)
mode = st.selectbox(
    label="mode",
    options=["📹 Video File", "🖼️ Image"],
    label_visibility="collapsed"
)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════════
if mode == "📹 Video File":
    st.markdown('<div class="section-label">Upload Video</div>', unsafe_allow_html=True)
    video_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if video_file:
        st.video(video_file)

        conf_threshold = 0.45  # fixed, no sidebar

        if st.button("▶  Run Detection"):
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output video
            out_path = tempfile.mktemp(suffix="_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0, text="Analysing video…")
            frame_placeholder = st.empty()

            total_persons_all = 0
            max_persons       = 0
            processed         = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, stream=True, verbose=False)
                person_count = 0

                for r in results:
                    for box in r.boxes:
                        cls  = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls == 0 and conf > conf_threshold:
                            person_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)

                            # Label background
                            label = f"Person {person_count}"
                            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 229, 255), -1)
                            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 12, 16), 1)

                # Count overlay (top-left)
                overlay_text = f"Persons Detected: {person_count}"
                cv2.rectangle(frame, (0, 0), (260, 42), (10, 12, 16), -1)
                cv2.putText(frame, overlay_text, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 229, 255), 2)

                out.write(frame)

                total_persons_all += person_count
                max_persons = max(max_persons, person_count)
                processed  += 1

                if processed % 8 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, use_container_width=True)
                    progress = min(processed / max(total_frames, 1), 1.0)
                    progress_bar.progress(progress, text=f"Frame {processed}/{total_frames}")

            cap.release()
            out.release()
            os.unlink(tfile.name)

            progress_bar.progress(1.0, text="Done!")
            frame_placeholder.empty()

            # ── Stats ────────────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            avg = round(total_persons_all / max(processed, 1), 1)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{processed}</div><div class="stat-label">Frames Processed</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{max_persons}</div><div class="stat-label">Peak Count</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{avg}</div><div class="stat-label">Avg per Frame</div></div>', unsafe_allow_html=True)

            # ── Download ─────────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Download Result</div>', unsafe_allow_html=True)
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇  Download Detected Video",
                    data=f,
                    file_name="human_detected.mp4",
                    mime="video/mp4"
                )
            os.unlink(out_path)

# ══════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if image_file:
        pil_img = Image.open(image_file).convert("RGB")
        st.image(pil_img, use_container_width=True, caption="Original")

        if st.button("▶  Run Detection"):
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            results = model(frame, verbose=False)

            person_count = 0
            for r in results:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > 0.45:
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                        label = f"Person {person_count}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 229, 255), -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 12, 16), 1)

            # Count overlay
            cv2.rectangle(frame, (0, 0), (260, 42), (10, 12, 16), -1)
            cv2.putText(frame, f"Persons Detected: {person_count}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 229, 255), 2)

            result_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.image(result_rgb, use_container_width=True, caption="Detection Result")

            st.markdown(f'<div class="stat-card" style="margin-top:1rem"><div class="stat-number">{person_count}</div><div class="stat-label">Persons Detected</div></div>', unsafe_allow_html=True)

            # Download
            st.markdown("<hr>", unsafe_allow_html=True)
            result_pil = Image.fromarray(result_rgb)
            import io
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇  Download Result Image",
                data=buf.getvalue(),
                file_name="human_detected.png",
                mime="image/png"
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#2d3340; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;">'
    'Human Detector · YOLOv8 · Ultralytics'
    '</div>',
    unsafe_allow_html=True
)
