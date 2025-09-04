import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Retail Heatmap Analytics", layout="wide")
st.title("Retail: Customer Movement & Heatmap Analysis")

uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    MODEL_WEIGHTS = "yolov8n.pt"
    CONF_THRESH = 0.3
    KERNEL_RADIUS = 25
    GAUSS_KSIZE = 51
    HEAT_DECAY = 0.98
    OVERLAY_ALPHA = 0.5
    CLASSES = [0]
    frame_skip = 2

    def generate_random_zones(frame_width, frame_height, num_zones=2, min_size_ratio=0.15, max_size_ratio=0.35):
        zones = {}
        for i in range(num_zones):
            w = random.randint(int(frame_width * min_size_ratio), int(frame_width * max_size_ratio))
            h = random.randint(int(frame_height * min_size_ratio), int(frame_height * max_size_ratio))
            x1 = random.randint(0, frame_width - w - 1)
            y1 = random.randint(0, frame_height - h - 1)
            x2 = x1 + w
            y2 = y1 + h
            zones[f"Zone{i+1}"] = (x1, y1, x2, y2)
        return zones

    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video.")
        st.stop()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    ZONES = generate_random_zones(width, height, num_zones=2)

    heat = np.zeros((height, width), dtype=np.float32)
    track_seen = set()
    frame_idx = 0

    occupancy_per_frame = []
    zone_occupancy = {z: [] for z in ZONES}
    dwell_frames = defaultdict(lambda: defaultdict(int))
    heat_intensity_zone = {z: 0.0 for z in ZONES}

    peak_frame = {"frame_idx": 0, "people": 0, "frame_image": None}

    st.info("Processing video...")

    # Preprocess frames into overlay_frames
    overlay_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        results = model.predict(frame, classes=CLASSES, conf=CONF_THRESH)
        frame_ids = set()
        frame_zone_count = {z: 0 for z in ZONES}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.arange(len(boxes))

            frame_ids.update(ids)
            track_seen.update(ids)

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                cv2.circle(heat, (cx, cy), KERNEL_RADIUS, 1, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)

                for z, (zx1, zy1, zx2, zy2) in ZONES.items():
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        frame_zone_count[z] += 1
                        dwell_frames[tid][z] += 1

        occupancy_per_frame.append(len(frame_ids))

        if len(frame_ids) > peak_frame["people"]:
            peak_frame["people"] = len(frame_ids)
            peak_frame["frame_idx"] = frame_idx
            peak_frame["frame_image"] = frame.copy()

        for z in ZONES:
            zone_occupancy[z].append(frame_zone_count[z])
            heat_intensity_zone[z] += frame_zone_count[z]

        heat *= HEAT_DECAY
        heat_blur = cv2.GaussianBlur(heat, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
        heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 1 - OVERLAY_ALPHA, heat_color, OVERLAY_ALPHA, 0)

        # Draw zones
        for z, (zx1, zy1, zx2, zy2) in ZONES.items():
            intensity = min(255, int(frame_zone_count[z] * 50))
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, intensity, 255), 3)

        overlay_frames.append(overlay)

    cap.release()
    st.success("Processing completed!")

    # Simulate live frames
    st.subheader("Live Overlay Simulation")
    frame_slot = st.empty()
    for frame in overlay_frames:
        frame_slot.image(frame, channels="BGR")
        time.sleep(1/fps)

    # Final heatmap
    heat_blur = cv2.GaussianBlur(heat, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    st.subheader("Final Heatmap")
    st.image(heat_color, channels="BGR")

    # Statistics
    st.subheader("Statistics")
    st.markdown(f"- Total unique people detected: **{len(track_seen)}**")
    st.markdown(f"- Frames processed: **{frame_idx}**")
    st.markdown(f"- Peak occupancy in any frame: **{peak_frame['people']}**")
    st.markdown(f"- Frame with most people: **{peak_frame['frame_idx']}**")
    st.markdown(f"- Average occupancy per frame: **{np.mean(occupancy_per_frame):.2f}**")

    if peak_frame["frame_image"] is not None:
        st.subheader("Frame with Peak Occupancy")
        st.image(peak_frame["frame_image"], channels="BGR")

    for z in ZONES:
        st.markdown(f"- {z} peak occupancy: {max(zone_occupancy[z], default=0)}, avg: {np.mean(zone_occupancy[z]):.2f}")
        occupancy_pct = (np.count_nonzero(zone_occupancy[z]) / len(zone_occupancy[z])) * 100
        st.markdown(f"- {z} occupancy percentage of frames: **{occupancy_pct:.1f}%**")
        avg_dwell_sec = (sum(dwell_frames[tid][z] for tid in dwell_frames) / fps) / len(dwell_frames) if dwell_frames else 0
        st.markdown(f"- {z} average dwell time per person: **{avg_dwell_sec:.2f} seconds**")
        st.markdown(f"- {z} cumulative heat intensity: **{heat_intensity_zone[z]:.1f}**")

        # Proof: occupancy over frames
        fig, ax = plt.subplots(figsize=(6,2))
        ax.plot(zone_occupancy[z], label=f"{z} occupancy")
        ax.set_xlabel("Frame")
        ax.set_ylabel("People count")
        ax.set_title(f"{z} Occupancy over Frames")
        ax.legend()
        st.pyplot(fig)
