"""Calibrate the gel-sensor marker grid and save x0/y0/dx/dy to a JSON file.

Workflow:
    python calibrate.py --camera 0
    # In PREVIEW mode you should see one green dot on each real marker.
    # Tune --lower / --upper / --denoise / --blob-min-area until detection is
    # clean, then press 'c'. The next 30 frames are accumulated into a union
    # bounding box and the result is saved to calibrations/<sensor>.json.

Hotkeys: q/ESC quit | c calibrate+save | r retry | m toggle mask
"""

import argparse
import time

import cv2

from calibration_io import (
    CalibrationAccumulator,
    CALIB_FRAMES,
    calibration_path,
    parse_bgr,
    parse_resize,
    save_calibration,
)
from src.tracker import find_markers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="0",
                   help="cv2 camera index (e.g. 0) or stream URL")
    p.add_argument("--resize", default="320x240",
                   help="WxH for client-side resize. 'off' for native size.")
    p.add_argument("--sensor", default="gelsight_mini",
                   help="profile name; saves to calibrations/<sensor>.json")
    p.add_argument("--N", type=int, default=7, help="grid rows (gelsight mini: 7)")
    p.add_argument("--M", type=int, default=9, help="grid cols (gelsight mini: 9)")
    p.add_argument("--lower", default="0,0,0", help="BGR lower bound for inRange")
    p.add_argument("--upper", default="100,110,140", help="BGR upper bound")
    p.add_argument("--blur", type=int, default=105, help="gaussian blur kernel (odd)")
    p.add_argument("--blur-sigma", type=float, default=15, help="gaussian sigma")
    p.add_argument("--border", type=int, default=0, help="zero this many edge px")
    p.add_argument("--denoise", type=int, default=5, help="median filter kernel (odd)")
    p.add_argument("--blob-min-area", type=int, default=5, help="SimpleBlobDetector minArea")
    p.add_argument("--calib-count-tolerance", type=float, default=0.10,
                   help="Frame-level rejection: skip frames whose detection "
                        "count deviates from N*M by more than this fraction "
                        "(default: 0.10 = ±10%). Pass 1.0 to accept all frames.")
    p.add_argument("--calib-percentile", type=float, default=2.0,
                   help="Point-level rejection: bbox is taken at the "
                        "[p, 100-p] percentiles of x/y across surviving frames "
                        "(default: 2.0). Pass 0 to use full min/max.")
    p.add_argument("--calib-frames", type=int, default=CALIB_FRAMES,
                   help=f"Frames to accumulate before saving (default: {CALIB_FRAMES})")
    return p.parse_args()


def main():
    args = parse_args()
    resize_to = parse_resize(args.resize)
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

    win = "marker_track calibrate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    bbox_acc = None
    show_mask = False
    saved_path = None

    def make_accumulator():
        return CalibrationAccumulator(
            expected_count=args.N * args.M,
            count_tolerance=args.calib_count_tolerance,
            percentile=args.calib_percentile,
        )

    fps_t0 = time.time()
    fps_n, fps = 0, 0.0
    first_frame = True

    print(f"[calibrate] target file: {calibration_path(args.sensor)}")
    print("[calibrate] q/ESC quit | c start calibrating | r retry | m toggle mask")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[calibrate] camera read failed")
            break

        src_h, src_w = frame.shape[:2]
        if resize_to is not None:
            frame = cv2.resize(frame, resize_to)
        h, w = frame.shape[:2]
        if first_frame:
            first_frame = False
            tag = f"-> {w}x{h}" if resize_to is not None else "(no resize)"
            print(f"[calibrate] source {src_w}x{src_h} {tag}")

        centers, mask = find_markers(frame, **detect_kwargs)

        if bbox_acc is not None and bbox_acc.frames < args.calib_frames:
            bbox_acc.update(centers)
            if bbox_acc.frames >= args.calib_frames:
                try:
                    x0, y0, dx, dy = bbox_acc.calibrate(args.N, args.M)
                    saved_path = save_calibration(
                        args.sensor, args.N, args.M, x0, y0, dx, dy)
                    print(f"[calibrate] kept {bbox_acc.frames} frames, "
                          f"rejected {bbox_acc.skipped}")
                    print(f"[calibrate] x0={x0:.1f} y0={y0:.1f} "
                          f"dx={dx:.2f} dy={dy:.2f}")
                    print(f"[calibrate] saved to {saved_path}")
                    for warn in bbox_acc.sanity_warnings(
                            args.N, args.M, x0, y0, dx, dy, w, h):
                        print(f"[calibrate] WARN: {warn}")
                except ValueError as e:
                    print(f"[calibrate] failed: {e}")
                    bbox_acc = None
                    saved_path = None

        if show_mask:
            view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            view = frame.copy()
        for x, y in centers:
            cv2.circle(view, (int(x), int(y)), 3, (0, 255, 0), -1)

        fps_n += 1
        if fps_n >= 10:
            now = time.time()
            fps = fps_n / (now - fps_t0)
            fps_t0 = now
            fps_n = 0

        if bbox_acc is not None and bbox_acc.frames < args.calib_frames:
            label = (f"CALIBRATING {bbox_acc.frames}/{args.calib_frames} "
                     f"(rejected {bbox_acc.skipped})")
        elif saved_path is not None:
            label = f"SAVED -> {saved_path.name}"
        else:
            label = "PREVIEW (press c)"
        hud = f"{label} | {w}x{h} | {fps:5.1f} fps | detected: {len(centers)}"
        cv2.putText(view, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('c'):
            if bbox_acc is None or saved_path is not None:
                bbox_acc = make_accumulator()
                saved_path = None
                print(f"[calibrate] accumulating over next {args.calib_frames} "
                      "frames — keep gel undeformed")
        if key == ord('r'):
            bbox_acc = make_accumulator()
            saved_path = None
            print("[calibrate] retrying calibration")
        if key == ord('m'):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
