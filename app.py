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
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* White background */
  .stApp {
    background-color: #ffffff;
    color: #1a1a1a;
  }

  /* Hide default header/footer/menu */
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero title */
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    font-size: 2.8rem;
    color: #1a1a1a;
    line-height: 1.1;
    margin-bottom: 0.3rem;
  }
  .hero-accent {
    color: #2563eb;
  }
  .hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: #9ca3af;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  /* Stat card */
  .stat-card {
    background: #f8f9fb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #2563eb;
    line-height: 1;
  }
  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    border: 1.5px dashed #d1d5db !important;
    border-radius: 12px !important;
    background: #f8f9fb !important;
    padding: 1rem !important;
  }

  /* Buttons */
  .stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
  }

  /* Download button */
  [data-testid="stDownloadButton"] > button {
    background: #f0f4ff !important;
    color: #2563eb !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: 1.5px solid #bfcfff !important;
    border-radius: 8px !important;
    width: 100% !important;
  }

  /* Progress bar */
  .stProgress > div > div {
    background-color: #2563eb !important;
  }

  /* Divider */
  hr {
    border-color: #e5e7eb;
    margin: 1.5rem 0;
  }

  /* Select box */
  .stSelectbox > div > div {
    background: #f8f9fb !important;
    border-color: #e5e7eb !important;
    color: #1a1a1a !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  /* Image caption */
  .stImage > div > div > div {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
  }

  /* Section label */
  .section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.5rem;
    font-weight: 500;
  }

  /* Tag badge */
  .badge {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin-right: 0.4rem;
    border: 1px solid #bfdbfe;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 0.5rem;">
  <div class="hero-title">Human <span class="hero-accent">Detection</span></div>

  <div style="font-family:'DM Sans',sans-serif; font-size:1rem; color:#4b5563; margin-top:0.6rem; margin-bottom:1.8rem;">Upload an image or video and detect humans instantly using YOLOv8.</div>
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
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (37, 99, 235), 2)

                            # Label background
                            label = f"Person {person_count}"
                            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (37, 99, 235), -1)
                            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # Count overlay (top-left)
                overlay_text = f"Persons Detected: {person_count}"
                cv2.rectangle(frame, (0, 0), (260, 42), (37, 99, 235), -1)
                cv2.putText(frame, overlay_text, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

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

            # ── Result Video on screen ────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

            # ── Download ─────────────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-top:1rem;">Download Result</div>', unsafe_allow_html=True)
            st.download_button(
                label="⬇  Download Detected Video",
                data=video_bytes,
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (37, 99, 235), 2)
                        label = f"Person {person_count}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (37, 99, 235), -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Count overlay
            cv2.rectangle(frame, (0, 0), (260, 42), (37, 99, 235), -1)
            cv2.putText(frame, f"Persons Detected: {person_count}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

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
    '<div style="text-align:center; color:#d1d5db; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase; font-family:\'DM Sans\',sans-serif;">'
    'Human Detector · YOLOv8 · Ultralytics'
    '</div>',
    unsafe_allow_html=True
)
