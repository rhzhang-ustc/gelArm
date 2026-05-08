# Marker Tracking Source Code

## Introduction

In this directory (./reactive_diffusion_policy/real_world/publisher/marker_track), we provide the source code and compilation files for find_marker.so (located at ./reactive_diffusion_policy/real_world/publisher/lib/find_marker.so). 

While our precompiled find_marker.so suffices for most scenarios, we provide the source code here to accommodate platform-specific compilation requirements and enable user customization.

## Requirements

* opencv
* pybind11
* numpy

```
pip3 install pybind11 numpy opencv-python
```

## Build from source

Enter the current directory

```
cd ./reactive_diffusion_policy/real_world/publisher/marker_track
```

Modify **makefile** based on your own platform

Make the project

```
make
```

Now **find_marker.so** should be under directory ./reactive_diffusion_policy/real_world/publisher/marker_track/lib, simply **replace** it with the original find_marker.so(located at ./reactive_diffusion_policy/real_world/publisher/lib/find_marker.so). 

## Python usage

After `make` produces `src/lib/find_marker*.so`, the full pipeline is available
through `src/tracker.py`. It exposes:

* `find_markers(frame, ...)` — detect marker centroids in a BGR frame.
* `MarkerTracker(N, M, fps, x0, y0, dx, dy)` — stateful matcher; the first
  call records rest positions, later calls return per-marker flow.
* `draw_flow(frame, result)` — overlay flow arrows for debugging.

```python
import cv2
from marker_track.src.tracker import MarkerTracker, find_markers, draw_flow

# Tune (x0, y0) to the pixel position of the top-left marker, and (dx, dy)
# to the pixel spacing between markers in your sensor's image.
tracker = MarkerTracker(N=8, M=8, fps=30, x0=80., y0=15., dx=21., dy=21.)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # One-shot: detect + match
    result = tracker.track(frame)

    # Or split if you want to inspect detection separately:
    # centers, mask = find_markers(frame)
    # result = tracker.match(centers)

    # result["flow"]      -> (N, M, 2) per-marker (dx, dy)
    # result["occupied"]  -> (N, M) int, -1 where the marker was inferred
    # result["Ox"/"Oy"]   -> (N, M) rest-grid positions
    # result["Cx"/"Cy"]   -> (N, M) current positions

    cv2.imshow("flow", draw_flow(frame, result))
    if cv2.waitKey(1) == ord('q'):
        break
```

The first frame establishes the rest grid, so capture an undeformed gel
image first; later frames produce non-zero flow when the gel deforms.

### Real-time pipeline

Two scripts split the work cleanly:

* **`calibrate.py`** — preview detection, then accumulate 30 frames and save
  `x0/y0/dx/dy/N/M` to `calibrations/<sensor>.json`.
* **`demo.py`** — load the saved calibration and track. Refuses to start if
  the JSON is missing.

**First run on a new sensor:**

```
cd marker_track
python calibrate.py --camera 0
```

You start in **PREVIEW** — green dots on each detected marker, count in the
HUD. Tune `--lower / --upper / --denoise / --blob-min-area` until exactly
one dot lands on every real marker. Then press `c`. The next 30 frames are
accumulated and calibration is computed via two-layer rejection:

1. **Frame-level rejection.** Frames whose detection count differs from
   `N*M` by more than `--calib-count-tolerance` (default ±10%) are skipped
   entirely. The HUD shows `CALIBRATING X/30 (rejected K)`.
2. **Point-level rejection.** Within surviving frames, the bbox is taken at
   the `[p, 100-p]` percentiles (default `--calib-percentile 2`), not
   min/max. A handful of stray points that slipped through stage 1 get
   trimmed.

After save, sanity warnings print if too many frames were rejected, if the
dx/dy ratio looks skewed, or if the bbox covers <30% of the frame (likely
missing corner markers). Press `r` to retry, `q`/ESC to quit.

The saved JSON looks like:

```json
{
  "N": 7, "M": 9,
  "x0": 12.5, "y0": 18.3,
  "dx": 33.4, "dy": 33.1
}
```

`N/M` default to **7×9** (gelsight mini). Use `--sensor <name>` to manage
multiple sensors. `x0/y0/dx/dy` come from the calibration only — they are
not CLI flags. `dx/dy` are required at runtime because the C++ matcher
uses them for distance pruning (`dmin = (dx*0.5)²`, `dmax = (dx*1.8)²`,
`moving_max = dx*2`), independent of where the rest positions land.

**Tracking — every later run:**

```
python demo.py --camera 0
```

The demo loads the calibration, detects, matches, and draws flow arrows.
No defensive filtering — assumes a sensor with reliable detection. If your
detection is unreliable, fix it at the source (tune `--lower` / `--upper`
/ `--blob-min-area` / `--denoise`) rather than papering over it downstream.

Hotkeys (demo): `q`/ESC quit, `r` reset rest grid (the next frame becomes
the new rest reference), `m` toggle the detection mask view.
Hotkeys (calibrate): `q`/ESC quit, `c` start the 30-frame calibration,
`r` retry, `m` toggle mask.

The HUD shows `WIDTHxHEIGHT | FPS | detected:N/expected`.

The masking pipeline (`tracker.py:find_markers`) does:

1. Subtract a wide Gaussian blur from the frame to flatten background
   illumination, then shift the residual so it sits in `[0, 255]` around
   `128`.
2. `cv2.inRange(diff, lower, upper)` keeps only BGR triples in the
   marker-shaped dark range. Defaults `(0,0,0)..(100,110,140)` come from
   the gelsight_marker_utils repo; lower the `upper` triple if the mask is
   too permissive (extra dots), raise it if real markers are missing.
3. Median filter (`--denoise`) removes isolated noise pixels.
4. `cv2.SimpleBlobDetector` extracts centroids — more robust on noisy
   masks than contour moments.

Each frame is resized to **320×240 by default** (`--resize 320x240`) before
detection + matching. Pass `--resize 640x480` for a different working size,
or `--resize off` to use the camera's native frame size — note that
calibration is resolution-specific, so re-save after changing this.

Run `python demo.py --help` for the full flag list.

## Third-party Components

The marker tracking component of this project is based on:

    [tracking] by Shaoxiong Wang

        Source: https://github.com/Gelsight/tracking

        License: MIT
