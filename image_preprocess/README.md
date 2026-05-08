# image_preprocess

Tools for viewing and capturing frames from the Raspberry Pi gel-sensor
camera.

## Layout

| File / folder | What it does |
|---|---|
| `streaming_from_raspberrypi.py` | Tkinter GUI: live stream on the left, **Save image** button on the right. Optional `--undistort` to apply a fisheye calibration before display. |
| `captures/` | Default save destination for the GUI. |
| [`camera_calibration/`](camera_calibration/) | One-time fisheye calibration. `calibrate_fisheye.py` captures checkerboard images (live preview with corner detection) and runs `cv2.fisheye.calibrate`, writing `fisheye_params.json` for the streaming tool to consume. |

Typical first-time setup: run `camera_calibration/calibrate_fisheye.py
capture` then `… calibrate` to produce `fisheye_params.json`, then run
`streaming_from_raspberrypi.py --undistort` for live-corrected frames.

## Raspberry Pi camera stream

A Raspberry Pi on the local network serves the gel-sensor camera as an MJPEG
stream at:

```
http://10.194.110.225:8000/index.html
```

### Browser (easiest)

Open the URL above in any browser on the same network. The page embeds the
live stream — use this for quick visual checks.

### OpenCV (for processing in Python)

`cv2.VideoCapture` reads MJPEG over HTTP directly. Point it at the raw
stream endpoint (`stream.mjpg`, not `index.html`):

```python
import cv2

STREAM_URL = "http://10.194.110.225:8000/stream.mjpg"

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise SystemExit("Could not open stream — check the Pi is reachable")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("pi cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### GUI capture tool (`streaming_from_raspberrypi.py`)

A small Tkinter app that shows the live stream on the left with a **Save
image** button on the right (spacebar also works). Each click writes the
current frame to `captures/capture_<timestamp>.png`.

Requires `pillow` in addition to `opencv-python`:

```
pip install pillow
```

Run:

```
cd image_preprocess
python streaming_from_raspberrypi.py
```

Defaults to `http://10.194.110.225:8000/stream.mjpg`; override with
`--stream <url>` for a different host. If you accidentally point it at the
viewer page (`/index.html`), it auto-swaps to `/stream.mjpg`.

To apply fisheye undistortion (using `camera_calibration/fisheye_params.json`):

```
python streaming_from_raspberrypi.py --undistort
```

Default is **off**. The save button writes whatever is currently displayed,
so saved frames are undistorted iff `--undistort` is on. Override the
calibration path with `--calibration <path>`.

## Fisheye camera calibration

[`camera_calibration/calibrate_fisheye.py`](camera_calibration/calibrate_fisheye.py)
calibrates a wide-FOV camera (e.g. the Pi Zero 160° fisheye) using a
checkerboard. Two-step workflow:

```
cd camera_calibration

# 1) Capture ~20+ board images at varied angles. Place the board near image
#    edges too — that's where fisheye distortion is strongest.
python calibrate_fisheye.py capture            # SPACE saves; q/ESC quits
python calibrate_fisheye.py capture --auto-save  # auto-save when board detected

# 2) Run calibration. Outputs fisheye_params.json next to the script.
python calibrate_fisheye.py calibrate --show   # --show previews undistortion
```

Defaults match an 8×11-square board with 20 mm squares (= 7×10 *inner*
corners). Override with `--pattern-size 7x10` and `--square-size 20`.

The output JSON contains the camera matrix `K`, distortion coefficients
`D`, and the image size used for calibration. To undistort a frame:

```python
import cv2, json, numpy as np
p = json.load(open("camera_calibration/fisheye_params.json"))
K, D = np.array(p["K"]), np.array(p["D"])
w, h = p["image_size"]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
```

Use `cv2.fisheye` (not `cv2.calibrateCamera`/`cv2.undistort`) — the pinhole
model breaks down past ~120° FOV.

## Use it with the marker tracker

`../marker_track/demo.py` accepts either a camera index or a stream URL via
`--camera`:

```
cd ../marker_track
python demo.py --camera http://10.194.110.225:8000/stream.mjpg
```
