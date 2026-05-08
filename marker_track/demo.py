"""Real-time marker tracking against a saved calibration.

Run `python calibrate.py --camera <src>` once first to produce a calibration
file. Then this demo loads it and just tracks.

Hotkeys: q/ESC quit | r reset rest grid | m toggle mask
"""

import argparse
import time

import cv2

from calibration_io import (
    calibration_path,
    load_calibration,
    parse_bgr,
    parse_resize,
)
from src.tracker import MarkerTracker, draw_flow, find_markers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="0",
                   help="cv2 camera index (e.g. 0) or stream URL")
    p.add_argument("--resize", default="320x240",
                   help="WxH for client-side resize. 'off' for native size.")
    p.add_argument("--sensor", default="gelsight_mini",
                   help="profile name; reads calibrations/<sensor>.json")
    p.add_argument("--fps", type=int, default=30,
                   help="matcher time budget (1/fps sec)")
    p.add_argument("--lower", default="0,0,0", help="BGR lower bound for inRange")
    p.add_argument("--upper", default="100,110,140", help="BGR upper bound")
    p.add_argument("--blur", type=int, default=105, help="gaussian blur kernel (odd)")
    p.add_argument("--blur-sigma", type=float, default=15, help="gaussian sigma")
    p.add_argument("--border", type=int, default=0, help="zero this many edge px")
    p.add_argument("--denoise", type=int, default=5, help="median filter kernel (odd)")
    p.add_argument("--blob-min-area", type=int, default=5,
                   help="SimpleBlobDetector minArea")
    return p.parse_args()


def main():
    args = parse_args()
    resize_to = parse_resize(args.resize)

    cal = load_calibration(args.sensor)
    if cal is None:
        raise SystemExit(
            f"[demo] no calibration at {calibration_path(args.sensor)}. "
            f"Run `python calibrate.py --camera <src> --sensor {args.sensor}` first."
        )
    N, M = cal["N"], cal["M"]
    print(f"[demo] loaded calibration from {calibration_path(args.sensor)}: "
          f"N={N} M={M} x0={cal['x0']:.1f} y0={cal['y0']:.1f} "
          f"dx={cal['dx']:.2f} dy={cal['dy']:.2f}")

    detect_kwargs = dict(
        lower=parse_bgr(args.lower),
        upper=parse_bgr(args.upper),
        blur_ksize=args.blur,
        blur_sigma=args.blur_sigma,
        border=args.border,
        denoise_ksize=args.denoise,
        blob_min_area=args.blob_min_area,
    )

    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source {args.camera!r}")

    def make_tracker():
        return MarkerTracker(
            N=N, M=M, fps=args.fps,
            x0=cal["x0"], y0=cal["y0"],
            dx=cal["dx"], dy=cal["dy"],
        )

    tracker = make_tracker()
    win = "marker_track demo"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    show_mask = False
    fps_t0 = time.time()
    fps_n, fps = 0, 0.0
    first_frame = True

    print("[demo] q/ESC quit | r reset rest grid | m toggle mask")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[demo] camera read failed")
            break

        src_h, src_w = frame.shape[:2]
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to)
        h, w = frame.shape[:2]
        if first_frame:
            first_frame = False
            tag = f"-> {w}x{h}" if resize_to is not None else "(no resize)"
            print(f"[demo] source {src_w}x{src_h} {tag}")

        centers, mask = find_markers(frame, **detect_kwargs)

        result = None
        if len(centers) > 0:
            result = tracker.match(centers)

        if show_mask:
            view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for x, y in centers:
                cv2.circle(view, (int(x), int(y)), 3, (0, 255, 0), -1)
        elif result is not None:
            view = draw_flow(frame, result)
        else:
            view = frame.copy()

        fps_n += 1
        if fps_n >= 10:
            now = time.time()
            fps = fps_n / (now - fps_t0)
            fps_t0 = now
            fps_n = 0

        hud = f"{w}x{h} | {fps:5.1f} fps | detected: {len(centers)}/{N*M}"
        cv2.putText(view, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            tracker = make_tracker()
            print("[demo] rest grid reset — next frame becomes the new reference")
        if key == ord('m'):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
