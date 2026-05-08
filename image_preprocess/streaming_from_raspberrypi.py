"""Stream the Raspberry Pi camera live and capture frames on demand.

Layout: live frame on the left, a "Save image" button (and saved-file
status) on the right. Click the button to write the current frame to
`captures/capture_<timestamp>.png`.

Requirements: opencv-python, pillow (`pip install pillow`).

Usage:
    python streaming_from_raspberrypi.py
    python streaming_from_raspberrypi.py --stream http://<host>:<port>/stream.mjpg
    python streaming_from_raspberrypi.py --save-dir my_captures
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk

DEFAULT_STREAM = "http://10.194.110.225:8000/stream.mjpg"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", default=DEFAULT_STREAM,
                   help=f"MJPEG stream URL (default: {DEFAULT_STREAM}). "
                        "Note: the Pi's /index.html is a browser page; cv2 "
                        "needs the raw /stream.mjpg endpoint.")
    p.add_argument("--save-dir", default="captures",
                   help="Subdirectory (under this script) to write captures into")
    p.add_argument("--display-fps", type=float, default=30.0,
                   help="GUI refresh rate (default: 30)")
    return p.parse_args()


def normalize_stream_url(url):
    """If user passed the viewer page, swap to the raw stream endpoint."""
    if url.endswith("/index.html"):
        fixed = url[: -len("index.html")] + "stream.mjpg"
        print(f"[stream] {url} is the viewer page — using {fixed}")
        return fixed
    return url


class FrameReader:
    """Background thread reading frames as fast as the source provides."""

    def __init__(self, url):
        self._url = url
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open stream {url!r}")
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = frame

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


def main():
    args = parse_args()
    stream_url = normalize_stream_url(args.stream)
    save_dir = Path(__file__).parent / args.save_dir
    save_dir.mkdir(exist_ok=True, parents=True)

    reader = FrameReader(stream_url)
    print(f"[stream] connected to {stream_url}")
    print(f"[stream] saves go to {save_dir}")

    root = tk.Tk()
    root.title("Raspberry Pi camera")

    img_label = tk.Label(root, bg="black", width=320, height=240)
    img_label.pack(side=tk.LEFT, padx=10, pady=10)

    side = tk.Frame(root)
    side.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    status_var = tk.StringVar(value="(no captures yet)")
    saved_count = {"n": 0}

    def on_save():
        frame = reader.latest()
        if frame is None:
            status_var.set("No frame yet")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"capture_{ts}.png"
        cv2.imwrite(str(path), frame)
        saved_count["n"] += 1
        status_var.set(f"Saved {path.name}\n({saved_count['n']} total)")
        print(f"[stream] saved {path}")

    save_btn = tk.Button(side, text="Save image", command=on_save,
                         height=2, width=15)
    save_btn.pack(pady=5)
    root.bind("<space>", lambda _e: on_save())

    tk.Label(side, textvariable=status_var,
             wraplength=200, justify=tk.LEFT).pack(pady=5)
    tk.Label(side, text="(spacebar also saves)", fg="gray").pack(pady=10)

    interval_ms = max(1, int(1000 / args.display_fps))

    def refresh():
        frame = reader.latest()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            img_label.config(image=photo, width=photo.width(),
                             height=photo.height())
            img_label.image = photo  # keep ref alive
        root.after(interval_ms, refresh)

    def on_close():
        reader.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(0, refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
