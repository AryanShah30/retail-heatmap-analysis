# Retail Heatmap Analysis

## Overview
This Streamlit app ingests shopper movement videos and produces per-frame heatmaps, occupancy metrics, and congestion alerts for retail spaces. It combines YOLOv8 detections with persistent multi-object tracking to follow people through the scene and aggregate their dwell time.

## Key Features
- Upload an MP4 clip directly in the browser.
- Live visualization of detections with a heatmap overlay.
- Interactive zone editor to add, update, or remove monitoring rectangles before processing.
- Optional floor plan upload so the final heatmap can be reprojected onto the actual store layout.
- Congestion flagging for dense scenes plus longest queue dwell stats using ByteTrack IDs and per-zone density thresholds.
- Top shopper insight that highlights the track ID with the highest cumulative dwell time and zone distribution.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit interface:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Upload a retail floor video (MP4). Use the sidebar zone editor to customise monitoring areas and optionally upload a floor plan; the sidebar also displays processing progress, live overlays, and congestion warnings when queues or bottlenecks form.

## Dense-Scene Handling
Dense areas such as checkout queues or gondola ends cause frequent occlusions that break naive frame-by-frame detections. The app now:
- Uses ByteTrack through Ultralytics YOLO to retain person IDs even under heavy overlap.
- Measures people-per-pixel density in each zone every frame.
- Highlights risk areas in red and reports the percentage of frames that were congested.
- Surfaces the longest queue dwell so staff can spot the slowest-moving lane or product hotspot.
- Calls out the top shopper so store teams can follow up on the highest dwell individual across all monitored zones.
- Counts zone occupancy using bounding box overlap so partial entries and queue edge cases are captured reliably.

These changes allow the heatmap and dwell-time analytics to stay stable even when shoppers are tightly clustered.
