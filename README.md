# gelArm

Working repo for the gelArim project: tooling around GelSight-style tactile
sensors streamed from a Raspberry Pi.

## Layout

| Folder | What it is |
|---|---|
| [`image_preprocess/`](image_preprocess/README.md) | Pi camera stream tools. Tkinter GUI (`streaming_from_raspberrypi.py`) for live view + save-image button, with optional fisheye undistortion. Fisheye calibration script under `camera_calibration/` produces the params used for undistortion. |
| [`marker_track/`](marker_track/README.md) | Real-time marker tracker for the gel surface. Wraps a fast C++ DFS matcher (`find_marker.so`, built from `src/tracking_class.cpp`) with Python helpers and two scripts: `calibrate.py` (one-time per sensor) and `demo.py` (live tracking + flow visualization). |

See the per-folder READMEs for setup, build instructions, and usage details.
