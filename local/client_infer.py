import argparse
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def pick_file_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        return path
    except Exception:
        return ""


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    p = argparse.ArgumentParser(description="Windows client: send image to server and save visualization.")
    p.add_argument("--server", default="http://127.0.0.1:18080", help="Server base URL (after port-forward).")
    p.add_argument("--input", default="", help="Input image path. If empty, use --pick.")
    p.add_argument("--pick", action="store_true", help="Open a file picker dialog to choose an image.")
    p.add_argument(
        "--out-dir",
        default=r"D:\pyCharmProjects\server\output",
        help=r"Output directory on Windows, default: D:\pyCharmProjects\server\output",
    )
    p.add_argument("--score-thresh", type=float, default=0.5, help="Optional score threshold override.")
    args = p.parse_args()

    image_path = args.input
    if args.pick or not image_path:
        image_path = pick_file_dialog()
    if not image_path:
        raise SystemExit("No input image selected.")

    img_bytes = Path(image_path).read_bytes()
    payload = {
        "image_b64": base64.b64encode(img_bytes).decode("utf-8"),
        "score_thresh": float(args.score_thresh),
    }

    resp = post_json(args.server.rstrip("/") + "/infer", payload)
    if not resp.get("ok", False):
        raise SystemExit(f"Server error: {resp.get('error')}")

    result = resp["result"]
    vis_bytes = base64.b64decode(result["vis_jpg_b64"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    out_img = out_dir / f"{stem}_vis.jpg"
    out_json = out_dir / f"{stem}_det.json"
    out_img.write_bytes(vis_bytes)
    out_json.write_text(json.dumps(result.get("detections", []), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_img}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()

