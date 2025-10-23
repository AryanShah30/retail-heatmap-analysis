# Retail Heatmap Analysis

## Overview
This Streamlit app ingests shopper movement videos and produces per-frame heatmaps, occupancy metrics, and congestion alerts for retail spaces. It combines YOLOv8 detections with persistent multi-object tracking to follow people through the scene and aggregate their dwell time.

## Key Features
- Upload an MP4 clip directly in the browser.
- Live visualization of detections with a heatmap overlay.
- Randomly generated monitoring zones with dwell-time statistics.
- Congestion flagging for dense scenes using ByteTrack IDs and per-zone density thresholds.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit interface:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Upload a retail floor video (MP4). The sidebar displays processing progress, live overlays, and congestion warnings when queues or bottlenecks form.

## Dense-Scene Handling
Dense areas such as checkout queues or gondola ends cause frequent occlusions that break naive frame-by-frame detections. The app now:
- Uses ByteTrack through Ultralytics YOLO to retain person IDs even under heavy overlap.
- Measures people-per-pixel density in each zone every frame.
- Highlights risk areas in red and reports the percentage of frames that were congested.

These changes allow the heatmap and dwell-time analytics to stay stable even when shoppers are tightly clustered.
