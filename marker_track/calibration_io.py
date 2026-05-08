"""Shared calibration I/O for calibrate.py and demo.py."""

import json
from pathlib import Path

CALIB_DIR = Path(__file__).parent / "calibrations"
CALIB_FRAMES = 30  # frames to accumulate before saving a calibration


def calibration_path(name):
    return CALIB_DIR / f"{name}.json"


def load_calibration(name):
    path = calibration_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_calibration(name, N, M, x0, y0, dx, dy):
    CALIB_DIR.mkdir(exist_ok=True)
    path = calibration_path(name)
    data = {
        "N": int(N), "M": int(M),
        "x0": round(float(x0), 3), "y0": round(float(y0), 3),
        "dx": round(float(dx), 3), "dy": round(float(dy), 3),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def parse_resize(s):
    if not s or s.lower() in ("off", "none", "0"):
        return None
    w, h = s.lower().split("x")
    return int(w), int(h)


def parse_bgr(s):
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"BGR triple must be 3 comma-separated ints, got {s!r}")
    return tuple(parts)


class CalibrationAccumulator:
    """Robust calibration via two-stage rejection.

    Stage 1 (frame rejection): skip any frame whose detection count
    deviates from N*M by more than ``count_tolerance``. A frame with way
    too many or too few detections almost certainly has noise/missing
    markers we don't want polluting the result.

    Stage 2 (point rejection): from the points in *surviving* frames,
    take the bbox at the [percentile, 100 - percentile] quantiles of
    each axis. This trims a handful of stray points that slipped past
    stage 1 (e.g., a single noise blob in an otherwise-clean frame).

    Both layers are configurable. Set ``count_tolerance=1`` to skip stage
    1 (accept all frames) or ``percentile=0`` to skip stage 2 (use full
    min/max of the surviving points).
    """
    def __init__(self, expected_count, count_tolerance=0.1, percentile=2.0):
        self.expected_count = expected_count
        self.count_tolerance = count_tolerance
        self.percentile = percentile
        self.xs = []
        self.ys = []
        self.frames = 0
        self.skipped = 0
        self.last_count = 0

    def update(self, centers):
        n = len(centers)
        self.last_count = n
        if n == 0:
            self.skipped += 1
            return
        deviation = abs(n - self.expected_count) / max(self.expected_count, 1)
        if deviation > self.count_tolerance:
            self.skipped += 1
            return
        self.xs.extend(centers[:, 0].tolist())
        self.ys.extend(centers[:, 1].tolist())
        self.frames += 1

    def total(self):
        return self.frames + self.skipped

    def calibrate(self, N, M):
        if self.frames < 5:
            raise ValueError(
                f"only {self.frames} usable frames out of {self.total()} "
                f"(expected ~{self.expected_count} markers/frame, "
                f"got {self.last_count} on the last frame). "
                "Fix detection (--lower/--upper/--blob-min-area) or relax "
                "--calib-count-tolerance, then retry."
            )
        import numpy as np
        xs = np.asarray(self.xs)
        ys = np.asarray(self.ys)
        p = self.percentile
        x0 = float(np.percentile(xs, p))
        y0 = float(np.percentile(ys, p))
        x1 = float(np.percentile(xs, 100 - p))
        y1 = float(np.percentile(ys, 100 - p))
        dx = (x1 - x0) / max(M - 1, 1)
        dy = (y1 - y0) / max(N - 1, 1)
        return x0, y0, dx, dy

    def sanity_warnings(self, N, M, x0, y0, dx, dy, frame_w, frame_h):
        msgs = []
        if self.skipped > self.frames:
            msgs.append(
                f"{self.skipped}/{self.total()} frames rejected "
                "(detection count mismatch). Tune detection or relax "
                "--calib-count-tolerance."
            )
        ratio = dx / dy if dy > 0 else float("inf")
        if not 0.7 < ratio < 1.4:
            msgs.append(
                f"dx/dy ratio = {ratio:.2f} (markers should be roughly "
                "square-spaced). A bad corner detection may be skewing the bbox."
            )
        if frame_w and frame_h:
            grid_area = (dx * (M - 1)) * (dy * (N - 1))
            coverage = grid_area / (frame_w * frame_h)
            if coverage < 0.3:
                msgs.append(
                    f"rest grid covers only {coverage:.0%} of the frame — "
                    "likely missing corner markers. Re-run with cleaner detection."
                )
        return msgs
