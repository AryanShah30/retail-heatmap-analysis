import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import random
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Heatmap Analytics", layout="wide")
st.title("Retail: Customer Movement & Heatmap Analysis")

uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    MODEL_WEIGHTS = "yolov8n.pt"
    CONF_THRESH = 0.3
    TRACKER_CONFIG = "bytetrack.yaml"
    KERNEL_RADIUS = 25
    GAUSS_KSIZE = 51
    HEAT_DECAY = 0.98
    OVERLAY_ALPHA = 0.5
    CLASSES = [0]
    frame_skip = 1
    CONGESTION_THRESHOLD = 1.2e-4  # people per pixel threshold to flag congestion
    MIN_INTERSECTION_RATIO = 0.2  # minimum bbox overlap with zone to count occupancy

    def generate_random_zones(frame_width, frame_height, num_zones=2, min_size_ratio=0.15, max_size_ratio=0.35):
        zones = []
        for i in range(num_zones):
            w = random.randint(int(frame_width * min_size_ratio), int(frame_width * max_size_ratio))
            h = random.randint(int(frame_height * min_size_ratio), int(frame_height * max_size_ratio))
            x1 = random.randint(0, frame_width - w - 1)
            y1 = random.randint(0, frame_height - h - 1)
            x2 = x1 + w
            y2 = y1 + h
            zones.append(
                {
                    "label": f"Zone{i+1}",
                    "coords": (x1, y1, x2, y2),
                }
            )
        return zones

    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video.")
        st.stop()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    if "zones" not in st.session_state or st.session_state.get("zones_dims") != (width, height):
        st.session_state["zones"] = generate_random_zones(width, height, num_zones=2)
        st.session_state["zones_dims"] = (width, height)

    def reset_zones():
        st.session_state["zones"] = generate_random_zones(width, height, num_zones=2)
        st.session_state["zones_dims"] = (width, height)

    floorplan_image = None

    with st.sidebar:
        st.subheader("Zone Management")
        st.caption("Define monitoring rectangles in pixel coordinates for this video.")
        st.write(f"Video resolution: {width}×{height}")

        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("Reset zones", key="reset_zones_button"):
                reset_zones()
                st.session_state["zone_select"] = "Add new zone"
                st.rerun()
        with ctrl_col2:
            if st.button("Clear floor plan", key="clear_floorplan_button"):
                st.session_state.pop("floorplan_bytes", None)
                st.rerun()

        floorplan_file = st.file_uploader("Optional floor plan image", type=["png", "jpg", "jpeg"], key="floorplan_uploader")
        if floorplan_file is not None:
            st.session_state["floorplan_bytes"] = floorplan_file.getvalue()

        stored_floorplan = st.session_state.get("floorplan_bytes")
        if stored_floorplan:
            np_bytes = np.asarray(bytearray(stored_floorplan), dtype=np.uint8)
            decoded_floorplan = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
            if decoded_floorplan is None:
                st.warning("Unable to read floor plan image. Upload a valid PNG or JPEG.")
                st.session_state.pop("floorplan_bytes", None)
            else:
                floorplan_image = decoded_floorplan
                st.image(cv2.cvtColor(floorplan_image, cv2.COLOR_BGR2RGB), caption="Floor plan preview", use_column_width=True)

        zones_list = st.session_state.get("zones", [])
        add_new_option = "Add new zone"

        if zones_list:
            zone_table = pd.DataFrame(
                [
                    {
                        "Zone": zone["label"],
                        "X1": zone["coords"][0],
                        "Y1": zone["coords"][1],
                        "X2": zone["coords"][2],
                        "Y2": zone["coords"][3],
                    }
                    for zone in zones_list
                ]
            )
            st.dataframe(zone_table, use_container_width=True, height=min(240, 60 * len(zones_list) + 38))
            selected = st.selectbox(
                "Edit an existing zone or add a new one",
                [zone["label"] for zone in zones_list] + [add_new_option],
                key="zone_select",
            )
        else:
            st.info("No zones defined. Use the form below to add your first zone.")
            selected = add_new_option
            st.session_state["zone_select"] = add_new_option

        editing_zone = next((z for z in zones_list if z["label"] == selected), None)
        suggested_label = editing_zone["label"] if editing_zone else f"Zone{len(zones_list) + 1}"

        if editing_zone and st.button("Remove selected zone", key="remove_selected_zone"):
            zones_list.remove(editing_zone)
            st.session_state["zone_select"] = add_new_option
            st.rerun()

        with st.form("zone_form"):
            st.markdown("Add or update a zone")
            label = st.text_input("Zone label", value=suggested_label)

            default_coords = (
                editing_zone["coords"]
                if editing_zone
                else (
                    0,
                    0,
                    min(max(0, width - 1), max(1, width // 3)),
                    min(max(0, height - 1), max(1, height // 3)),
                )
            )
            c1, c2 = st.columns(2)
            with c1:
                x1 = st.number_input("X1 (left)", min_value=0, max_value=max(0, width - 1), value=int(default_coords[0]), step=1)
                y1 = st.number_input("Y1 (top)", min_value=0, max_value=max(0, height - 1), value=int(default_coords[1]), step=1)
            with c2:
                x2 = st.number_input("X2 (right)", min_value=0, max_value=max(0, width - 1), value=int(default_coords[2]), step=1)
                y2 = st.number_input("Y2 (bottom)", min_value=0, max_value=max(0, height - 1), value=int(default_coords[3]), step=1)

            submitted = st.form_submit_button("Save zone")
            if submitted:
                zone_label = label.strip() or suggested_label
                if x2 <= x1 or y2 <= y1:
                    st.warning("X2 must be greater than X1 and Y2 must be greater than Y1.")
                else:
                    zone_coords = (int(x1), int(y1), int(x2), int(y2))
                    duplicate = next((z for z in zones_list if z["label"] == zone_label), None)

                    if editing_zone:
                        if duplicate and duplicate is not editing_zone:
                            st.warning("Zone label already exists. Choose a different label.")
                        else:
                            editing_zone["label"] = zone_label
                            editing_zone["coords"] = zone_coords
                            st.session_state["zone_select"] = zone_label
                            st.rerun()
                    else:
                        if duplicate:
                            st.warning("Zone label already exists. Choose a different label.")
                        else:
                            zones_list.append({"label": zone_label, "coords": zone_coords})
                            st.session_state["zone_select"] = zone_label
                            st.rerun()

    ZONES = {zone["label"]: zone["coords"] for zone in st.session_state.get("zones", [])}

    heat = np.zeros((height, width), dtype=np.float32)
    track_seen = set()
    frame_idx = 0

    occupancy_per_frame = []
    zone_occupancy = {z: [] for z in ZONES}
    dwell_frames = defaultdict(lambda: defaultdict(int))
    heat_intensity_zone = {z: 0.0 for z in ZONES}
    zone_density_history = {z: [] for z in ZONES}
    zone_congestion_frames = {z: 0 for z in ZONES}

    peak_frame = {"frame_idx": 0, "people": 0, "frame_image": None}

    st.info("Processing video...")
    progress_text = st.empty()
    frame_slot = st.empty()
    congestion_banner = st.empty()

    try:
        results_stream = model.track(
            source=video_path,
            stream=True,
            tracker=TRACKER_CONFIG,
            persist=True,
            classes=CLASSES,
            conf=CONF_THRESH,
            verbose=False,
        )
    except Exception as err:
        st.error(f"Tracking initialization failed: {err}")
        st.stop()

    for result in results_stream:
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = result.orig_img.copy()
        frame_ids = set()
        frame_zone_count = {z: 0 for z in ZONES}
        congested_zones = []

        boxes_tensor = getattr(result, "boxes", None)
        if boxes_tensor is not None and boxes_tensor.xyxy is not None:
            boxes = boxes_tensor.xyxy.cpu().numpy().astype(int)
            if boxes_tensor.id is not None:
                ids = boxes_tensor.id.cpu().numpy().astype(int)
            else:
                ids = np.arange(len(boxes))

            frame_ids.update(ids)
            track_seen.update(int(tid) for tid in ids)

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                cv2.circle(heat, (cx, cy), KERNEL_RADIUS, 1, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
                cv2.putText(frame, f"ID {int(tid)}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)

                bbox_area = max(1, (x2 - x1) * (y2 - y1))

                for z, (zx1, zy1, zx2, zy2) in ZONES.items():
                    inter_x1 = max(x1, zx1)
                    inter_y1 = max(y1, zy1)
                    inter_x2 = min(x2, zx2)
                    inter_y2 = min(y2, zy2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    if inter_area and inter_area / bbox_area >= MIN_INTERSECTION_RATIO:
                        frame_zone_count[z] += 1
                        dwell_frames[int(tid)][z] += 1

        occupancy_per_frame.append(len(frame_ids))

        if len(frame_ids) > peak_frame["people"]:
            peak_frame["people"] = len(frame_ids)
            peak_frame["frame_idx"] = frame_idx
            peak_frame["frame_image"] = frame.copy()

        for z in ZONES:
            zone_occupancy[z].append(frame_zone_count[z])
            heat_intensity_zone[z] += frame_zone_count[z]
            zx1, zy1, zx2, zy2 = ZONES[z]
            zone_area = max(1, (zx2 - zx1) * (zy2 - zy1))
            density = frame_zone_count[z] / zone_area
            zone_density_history[z].append(density)
            if density >= CONGESTION_THRESHOLD:
                zone_congestion_frames[z] += 1
                congested_zones.append(z)

        heat *= HEAT_DECAY
        heat_blur = cv2.GaussianBlur(heat, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
        heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 1 - OVERLAY_ALPHA, heat_color, OVERLAY_ALPHA, 0)

        for z, (zx1, zy1, zx2, zy2) in ZONES.items():
            intensity = min(255, int(frame_zone_count[z] * 50))
            color = (0, intensity, 255)
            if z in congested_zones:
                color = (0, 0, 255)
                cv2.putText(
                    overlay,
                    "Queue risk",
                    (zx1 + 5, min(height - 10, zy1 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, 3)

        frame_slot.image(overlay, channels="BGR")
        if congested_zones:
            congestion_banner.warning(f"High density in: {', '.join(congested_zones)}")
        else:
            congestion_banner.empty()

        if total_frames:
            progress_text.text(f"Processed frame {frame_idx}/{total_frames}")
        else:
            progress_text.text(f"Processed frame {frame_idx}")

    st.success("Processing completed!")

    heat_blur = cv2.GaussianBlur(heat, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    st.subheader("Final Heatmap")
    st.image(heat_color, channels="BGR")

    if floorplan_image is not None:
        if floorplan_image.shape[0] != height or floorplan_image.shape[1] != width:
            aligned_floorplan = cv2.resize(floorplan_image, (width, height))
        else:
            aligned_floorplan = floorplan_image
        floorplan_overlay = cv2.addWeighted(aligned_floorplan, 1 - OVERLAY_ALPHA, heat_color, OVERLAY_ALPHA, 0)
        st.subheader("Heatmap on Floor Plan")
        st.image(floorplan_overlay, channels="BGR")

    per_zone_longest = {z: {"seconds": 0.0, "track_id": None} for z in ZONES}
    longest_queue = {"zone": None, "seconds": 0.0, "track_id": None}
    top_shopper = {"track_id": None, "seconds": 0.0, "frames_by_zone": {}}
    for tid, zone_map in dwell_frames.items():
        total_frames_tid = sum(zone_map.values())
        if total_frames_tid > 0:
            total_seconds = total_frames_tid / fps if fps else 0
            if total_seconds > top_shopper["seconds"]:
                top_shopper = {
                    "track_id": tid,
                    "seconds": total_seconds,
                    "frames_by_zone": {zone: frames for zone, frames in zone_map.items() if frames > 0},
                }
        for z, frames_count in zone_map.items():
            if z not in per_zone_longest or frames_count <= 0:
                continue
            seconds = frames_count / fps if fps else 0
            if seconds > per_zone_longest[z]["seconds"]:
                per_zone_longest[z] = {"seconds": seconds, "track_id": tid}
            if seconds > longest_queue["seconds"]:
                longest_queue = {"zone": z, "seconds": seconds, "track_id": tid}

    st.subheader("Statistics")
    st.markdown(f"- Total unique people detected: **{len(track_seen)}**")
    st.markdown(f"- Frames processed: **{frame_idx}**")
    st.markdown(f"- Peak occupancy in any frame: **{peak_frame['people']}** (unique IDs)")
    st.write("")
    st.markdown(f"- Frame with most people: **{peak_frame['frame_idx']}**")
    avg_occupancy = np.mean(occupancy_per_frame) if occupancy_per_frame else 0
    st.markdown(f"- Average occupancy per frame: **{avg_occupancy:.2f}**")
    if longest_queue["zone"] is not None:
        st.markdown(
            f"- Longest queue dwell: **{longest_queue['seconds']:.2f} seconds** in {longest_queue['zone']} (ID {longest_queue['track_id']})"
        )
    if top_shopper["track_id"] is not None:
        st.markdown(
            f"- Top shopper dwell: **{top_shopper['seconds']:.2f} seconds** (ID {top_shopper['track_id']})"
        )
        if top_shopper["frames_by_zone"]:
            contribution_df = pd.DataFrame(
                [
                    {
                        "Zone": zone,
                        "Frames": frames,
                        "Dwell (s)": frames / fps if fps else 0,
                    }
                    for zone, frames in top_shopper["frames_by_zone"].items()
                ]
            ).sort_values("Dwell (s)", ascending=False)
            contribution_df["Frames"] = contribution_df["Frames"].astype(int)
            contribution_df["Dwell (s)"] = contribution_df["Dwell (s)"].round(2)
            st.caption("Top shopper dwell distribution by zone")
            st.table(contribution_df.reset_index(drop=True))

    if peak_frame["frame_image"] is not None:
        st.subheader("Frame with Peak Occupancy")
        st.image(peak_frame["frame_image"], channels="BGR")

    for z in ZONES:
        st.write("")
        avg_zone_occupancy = np.mean(zone_occupancy[z]) if zone_occupancy[z] else 0
        st.markdown(f"- {z} peak occupancy: {max(zone_occupancy[z], default=0)}, avg: {avg_zone_occupancy:.2f}")
        occupancy_pct = (np.count_nonzero(zone_occupancy[z]) / len(zone_occupancy[z]) * 100) if zone_occupancy[z] else 0
        st.markdown(f"- {z} occupancy percentage of frames: **{occupancy_pct:.1f}%**")
        zone_dwell_frames = [zone_map.get(z, 0) for zone_map in dwell_frames.values() if zone_map.get(z, 0)]
        avg_dwell_sec = (sum(zone_dwell_frames) / fps / len(zone_dwell_frames)) if zone_dwell_frames else 0
        st.markdown(f"- {z} average dwell time per person: **{avg_dwell_sec:.2f} seconds**")
        st.markdown(f"- {z} cumulative heat intensity: **{heat_intensity_zone[z]:.1f}**")
        congested_pct = (zone_congestion_frames[z] / len(zone_density_history[z]) * 100) if zone_density_history[z] else 0
        st.markdown(f"- {z} congestion frames: **{zone_congestion_frames[z]} ({congested_pct:.1f}% of processed frames)**")
        zone_longest = per_zone_longest.get(z, {"seconds": 0.0, "track_id": None})
        if zone_longest["track_id"] is not None:
            st.markdown(
                f"- {z} longest dwell: **{zone_longest['seconds']:.2f} seconds** (ID {zone_longest['track_id']})"
            )
        else:
            st.markdown(f"- {z} longest dwell: **0.00 seconds**")

        fig, ax = plt.subplots(figsize=(6,2))
        ax.plot(zone_occupancy[z], label=f"{z} occupancy")
        ax.set_xlabel("Frame")
        ax.set_ylabel("People count")
        ax.set_title(f"{z} Occupancy over Frames")
        ax.legend()
        st.pyplot(fig)
