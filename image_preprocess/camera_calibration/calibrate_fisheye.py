"""Fisheye camera calibration using a checkerboard.

For wide-FOV cameras (here: Raspberry Pi Zero 160°). Uses the dedicated
``cv2.fisheye`` model — ``cv2.calibrateCamera``'s pinhole model breaks down
past ~120° FOV.

Default pattern: an 8×11-square checkerboard with 20 mm squares
(= 7×10 *inner* corners). Override with ``--pattern-size`` and
``--square-size``.

Two-step workflow:

1. Capture ~20+ images of the board at varied angles and positions.
   Position the board near image *edges* too — that's where fisheye
   distortion is strongest and where the model needs the most data.

       python calibrate_fisheye.py capture

   Live view shows the detected corners; press SPACE to save the current
   frame, ``q``/ESC to quit. ``--auto-save`` saves automatically (~1.5 s
   apart) whenever the board is detected.

2. Run calibration on the saved images:

       python calibrate_fisheye.py calibrate --show

   Writes ``fisheye_params.json`` next to this script (camera matrix
   ``K``, distortion coefficients ``D``, image size, RMS reprojection
   error). ``--show`` previews before/after undistortion on one image.

Usage from another script:

    import json, cv2, numpy as np
    p = json.load(open("fisheye_params.json"))
    K = np.array(p["K"]); D = np.array(p["D"])
    h, w = p["image_size"][1], p["image_size"][0]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_STREAM = "http://10.194.110.225:8000/stream.mjpg"
HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture",
                         help="capture checkerboard images interactively")
    cap.add_argument("--stream", default=DEFAULT_STREAM,
                     help=f"MJPEG URL or camera index (default: {DEFAULT_STREAM})")
    cap.add_argument("--image-dir", default="captures",
                     help="folder under camera_calibration/ to save into")
    cap.add_argument("--pattern-size", default="7x10",
                     help="inner corners cols x rows (default 7x10 for an "
                          "8x11-square board)")
    cap.add_argument("--auto-save", action="store_true",
                     help="auto-save when the board is detected (1.5s apart)")

    cal = sub.add_parser("calibrate",
                         help="run fisheye calibration on saved images")
    cal.add_argument("--image-dir", default="captures",
                     help="folder of checkerboard images")
    cal.add_argument("--pattern-size", default="7x10",
                     help="inner corners cols x rows")
    cal.add_argument("--square-size", type=float, default=20.0,
                     help="square edge length in mm (default: 20)")
    cal.add_argument("--output", default="fisheye_params.json",
                     help="output JSON filename")
    cal.add_argument("--show", action="store_true",
                     help="show before/after undistortion on the first image")

    return p.parse_args()


def parse_pattern(s):
    cols, rows = s.lower().split("x")
    return int(cols), int(rows)


def open_source(stream):
    src = int(stream) if stream.isdigit() else stream
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {stream!r}")
    return cap


def capture_mode(args):
    pattern = parse_pattern(args.pattern_size)
    save_dir = HERE / args.image_dir
    save_dir.mkdir(exist_ok=True, parents=True)

    cap = open_source(args.stream)
    print(f"[capture] saving to {save_dir}")
    print("[capture] SPACE save | q/ESC quit"
          + (" | auto-save ON" if args.auto_save else ""))

    saved = 0
    last_auto = 0.0
    win = "calibrate-capture"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[capture] read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)

        status = "FOUND" if found else "no checkerboard"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(view, f"{status} | saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, f"{status} | saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        save = (key == 32)  # space
        if key in (ord('q'), 27):
            break
        if args.auto_save and found and time.time() - last_auto > 1.5:
            save = True
            last_auto = time.time()

        if save:
            if not found:
                print("[capture] no checkerboard — not saving")
                continue
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"checker_{ts}_{saved:03d}.png"
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"[capture] saved {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[capture] {saved} images saved to {save_dir}")


def calibrate_mode(args):
    pattern = parse_pattern(args.pattern_size)
    cols, rows = pattern
    img_dir = HERE / args.image_dir
    images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    if not images:
        raise SystemExit(f"No .png/.jpg in {img_dir}")

    # 3D object points: same checkerboard layout, in mm. Shape (1, N, 3) per
    # image — that's what cv2.fisheye.calibrate expects.
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_size

    obj_points = []
    img_points = []
    img_size = None
    used = []
    skipped = []

    refine_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            skipped.append((path.name, "unreadable"))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)
        elif (gray.shape[1], gray.shape[0]) != img_size:
            skipped.append((path.name, "size mismatch"))
            continue

        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            skipped.append((path.name, "no corners"))
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), refine_criteria)

        obj_points.append(objp.copy())
        img_points.append(corners.reshape(1, -1, 2))
        used.append(path.name)

    if len(obj_points) < 5:
        raise SystemExit(
            f"Need >=5 valid images, got {len(obj_points)}. Skipped: {skipped}"
        )

    print(f"[calibrate] using {len(used)}/{len(images)} images, "
          f"size {img_size}")
    if skipped:
        print(f"[calibrate] skipped {len(skipped)} (showing first 5):")
        for name, why in skipped[:5]:
            print(f"  - {name}: {why}")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_points]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_points]

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
             | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_CHECK_COND)

    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_points, img_points, img_size, K, D, rvecs, tvecs, flags,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    print(f"[calibrate] RMS reprojection error: {rms:.3f} px")
    print(f"[calibrate] K =\n{K}")
    print(f"[calibrate] D = {D.flatten()}")

    out = {
        "image_size": [int(img_size[0]), int(img_size[1])],  # [w, h]
        "K": K.tolist(),
        "D": D.flatten().tolist(),
        "pattern_size": [int(cols), int(rows)],
        "square_size_mm": float(args.square_size),
        "n_images_used": len(used),
        "rms_error_px": float(rms),
    }
    out_path = HERE / args.output
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[calibrate] saved to {out_path}")

    if args.show:
        first = cv2.imread(str(img_dir / used[0]))
        h, w = first.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
        undist = cv2.remap(first, map1, map2,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        side = np.hstack([first, undist])
        cv2.putText(side, "original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(side, "undistorted", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("original | undistorted", side)
        print("[calibrate] press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.mode == "capture":
        capture_mode(args)
    else:
        calibrate_mode(args)


if __name__ == "__main__":
    main()
