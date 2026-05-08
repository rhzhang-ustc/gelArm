"""Python wrappers for the marker_track C++ pipeline.

Pipeline:
    frame ─▶ find_markers()  ─▶ centers (Nx2) ─▶ MarkerTracker.match() ─▶ flow

Run `make` in marker_track/ first to build src/lib/find_marker*.so.
"""

import os
import sys
import numpy as np
import cv2

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

try:
    import find_marker as _fm
except ImportError as e:
    raise ImportError(
        f"Could not import find_marker from {_LIB_DIR}. "
        "Build it first: cd marker_track && make"
    ) from e


def find_markers(
    frame,
    lower=(0, 0, 0),
    upper=(100, 110, 140),
    blur_ksize=105,
    blur_sigma=15,
    border=0,
    denoise_ksize=5,
    blob_min_area=5,
):
    """Detect marker blobs using a BGR-range mask + SimpleBlobDetector.

    Pipeline (adapted from gelsight_marker_utils.get_tactile_mask):

    1. Estimate the background illumination with a large Gaussian blur.
    2. Normalize the residual `frame - blur` into [0, 255], shifted so that
       a zero residual maps to 128. This makes both darker-than-background
       and brighter-than-background pixels measurable in a single pass.
    3. ``cv2.inRange`` with BGR ``lower``/``upper`` keeps marker-shaped dark
       residuals while rejecting bright spurious detections that a plain
       single-channel threshold would let through.
    4. Optionally zero out an outer ``border`` (helps if the camera sees a
       bezel around the gel; default 0 because most streams fill the frame).
    5. Median filter on the binary mask kills isolated noise pixels.
    6. ``cv2.SimpleBlobDetector`` extracts blob centroids (more robust on
       noisy/irregular blobs than ``findContours`` + moments).

    Args:
        frame:          BGR image.
        lower, upper:   BGR bounds for ``cv2.inRange`` on the shifted diff.
        blur_ksize:     Gaussian kernel size for background estimation (odd).
        blur_sigma:     Gaussian sigma (0 → derived from kernel).
        border:         Px to zero out on each side of the mask.
        denoise_ksize:  Median filter kernel for the binary mask (odd).
        blob_min_area:  ``SimpleBlobDetector.minArea``.

    Returns:
        centers: ``(K, 2)`` float array of ``(x, y)`` marker centroids.
        mask:    binary ``uint8`` mask of detected marker pixels.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if denoise_ksize % 2 == 0:
        denoise_ksize += 1

    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), blur_sigma)
    diff = frame.astype(np.float64) / 255.0 - blurred.astype(np.float64) / 255.0
    diff = 255.0 * (diff + 0.5)
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    mask = cv2.inRange(diff, np.asarray(lower, dtype=np.uint8),
                       np.asarray(upper, dtype=np.uint8))

    if border > 0:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0

    if denoise_ksize > 1:
        mask = cv2.medianBlur(mask, denoise_ksize)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = float(blob_min_area)
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(255 - mask)
    centers = np.array([kp.pt for kp in keypoints],
                       dtype=np.float64).reshape(-1, 2)
    return centers, mask


class MarkerTracker:
    """Stateful tracker — wraps find_marker.Matching.

    The first call to `match` (or `track`) records rest positions of the
    grid. Subsequent calls return per-marker flow against that rest grid.
    """

    def __init__(self, N=8, M=8, fps=30, x0=80.0, y0=15.0, dx=21.0, dy=21.0):
        self.N, self.M = N, M
        self._matcher = _fm.Matching(N, M, fps, x0, y0, dx, dy)

    def match(self, centers):
        """Match a set of detected centers to the NxM grid.

        Args:
            centers: (K, 2) array-like of (x, y) detections.
        Returns:
            dict with keys:
                Ox, Oy:     (N, M) rest grid positions
                Cx, Cy:     (N, M) current positions (inferred where occluded)
                occupied:   (N, M) int — index of detection assigned, or -1
                flow:       (N, M, 2) array of (dx, dy) per slot
        """
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        self._matcher.init(centers.tolist())
        self._matcher.run()
        Ox, Oy, Cx, Cy, occ = self._matcher.get_flow()
        Ox = np.asarray(Ox); Oy = np.asarray(Oy)
        Cx = np.asarray(Cx); Cy = np.asarray(Cy)
        occ = np.asarray(occ, dtype=np.int32)
        flow = np.stack([Cx - Ox, Cy - Oy], axis=-1)
        return {"Ox": Ox, "Oy": Oy, "Cx": Cx, "Cy": Cy,
                "occupied": occ, "flow": flow}

    def track(self, frame, **detect_kwargs):
        """Detect markers in `frame` and match them in one call."""
        centers, mask = find_markers(frame, **detect_kwargs)
        result = self.match(centers)
        result["centers"] = centers
        result["mask"] = mask
        return result


def draw_flow(frame, result, scale=2.0, color=(0, 0, 255),
              inferred_color=(127, 127, 255), thickness=2):
    """Overlay flow arrows on a frame (mirrors the C++ visualization)."""
    out = frame.copy()
    Ox, Oy = result["Ox"], result["Oy"]
    Cx, Cy = result["Cx"], result["Cy"]
    occ = result["occupied"]
    N, M = Ox.shape
    for i in range(N):
        for j in range(M):
            a = (int(Ox[i, j]), int(Oy[i, j]))
            b = (int(Cx[i, j] + scale * (Cx[i, j] - Ox[i, j])),
                 int(Cy[i, j] + scale * (Cy[i, j] - Oy[i, j])))
            c = inferred_color if occ[i, j] <= -1 else color
            cv2.arrowedLine(out, a, b, c, thickness)
    return out
