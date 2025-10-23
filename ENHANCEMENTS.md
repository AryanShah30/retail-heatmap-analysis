# Enhancement Notes

## Context
The initial release of `streamlit_app.py` relied solely on frame-by-frame YOLO detections. When shoppers crowded into narrow zones (checkout queues, gondola ends) the detector lost people due to occlusions, instantly dropping their contributions to the heatmap and dwell timers. There was no signal to warn operators that a zone was oversubscribed.

## What Has Changed
- **Persistent Tracking:** The app now runs Ultralytics YOLO in `track` mode with ByteTrack (`bytetrack.yaml`). IDs are kept across frames, allowing us to follow each person through occlusion-heavy scenes and maintain heatmap intensity.
- **Density-Based Congestion Metrics:** For every zone, the pipeline computes people-per-pixel density each frame. When a density crosses `CONGESTION_THRESHOLD`, the zone is marked congested, tinted red, and a live “Queue risk” banner is shown.
- **Post-Run Congestion Summary:** Zone-level statistics now report how many frames (and percentage) were congested, helping operators measure how often bottlenecks occur.
- **Interactive Zone Management:** Operators can now add, update, or remove monitoring rectangles from the sidebar before analysis, replacing the earlier random layout generator.
- **Longest Queue Dwell Insight:** Statistics call out the track ID and zone with the highest per-person dwell time so teams can pinpoint the slowest-moving queue or product area.
- **Floor Plan Overlay:** A floor plan image can be uploaded and the final heatmap will be reprojected onto it, giving teams a layout-accurate view of congestion.
- **Top Shopper Highlight:** The analytics surface the shopper ID that spent the most time in-store and break down their dwell by zone.
- **Overlap-Based Zone Counting:** Zone occupancy now uses bounding box overlap rather than only the detection centroid, improving accuracy for queues that graze zone boundaries.

## Resulting Impact
The updated tracking and density checks close the previously identified gap: dense scenes are no longer misinterpreted as low traffic, and store teams receive actionable warnings when space utilization becomes risky. Interactive zones keep analytics aligned with the actual floor layout, while longest-queue dwell stats highlight the shoppers that wait the longest. Floor plan overlays and the top-shopper insight make it easier to translate heatmap findings into staffing or merchandising actions. The heatmap analytics remain stable, and dwell time reporting stays accurate even under crowding.
