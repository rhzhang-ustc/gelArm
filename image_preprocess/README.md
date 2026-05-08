# image_preprocess

Tools for viewing and capturing frames from the Raspberry Pi gel-sensor
camera.

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

## Use it with the marker tracker

`../marker_track/demo.py` accepts either a camera index or a stream URL via
`--camera`:

```
cd ../marker_track
python demo.py --camera http://10.194.110.225:8000/stream.mjpg
```
