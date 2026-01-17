import argparse
import base64
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def pick_file_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
    except Exception:
        return ""


def post_json(url: str, payload: dict, token: str, timeout_s: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json", "X-Relay-Token": token})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, token: str, timeout_s: int = 120) -> dict:
    req = Request(url, headers={"X-Relay-Token": token})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args():
    p = argparse.ArgumentParser(description="Local PC submitter via VPS relay (Scheme A).")
    p.add_argument("--vps", default="http://47.237.1.93:18081", help="VPS relay base url.")
    p.add_argument("--token", required=True, help="Relay token (X-Relay-Token).")
    p.add_argument("--input", default="", help="Input image path. If empty, use --pick.")
    p.add_argument("--pick", action="store_true", help="Choose image via dialog.")
    p.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold.")
    p.add_argument("--out-dir", default=r"D:\pyCharmProjects\server\output", help="Where to save outputs.")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between polls.")
    return p.parse_args()


def main():
    args = parse_args()
    vps = args.vps.rstrip("/")
    token = args.token.strip().strip("'\"")

    image_path = args.input
    if args.pick or not image_path:
        image_path = pick_file_dialog()
    if not image_path:
        raise SystemExit("No input image selected.")

    img_bytes = Path(image_path).read_bytes()
    payload = {
        "filename": Path(image_path).name,
        "image_b64": base64.b64encode(img_bytes).decode("utf-8"),
        "score_thresh": float(args.score_thresh),
    }

    resp = post_json(vps + "/api/v1/jobs", payload, token=token, timeout_s=120)
    if not resp.get("ok", False):
        raise SystemExit(f"Relay error: {resp.get('error')}")
    job_id = resp["job_id"]
    print(f"Submitted job_id={job_id}")

    while True:
        r = get_json(vps + f"/api/v1/jobs/{job_id}", token=token, timeout_s=120)
        if not r.get("ok", False):
            raise SystemExit(f"Relay error: {r.get('error')}")
        job = r["job"]
        status = job["status"]
        if status in ("queued", "processing"):
            print(f"status={status}")
            time.sleep(args.poll_interval)
            continue
        if status == "failed":
            raise SystemExit(f"Job failed: {job.get('error')}")
        if status == "done":
            result = job.get("result") or {}
            vis_b64 = result.get("vis_jpg_b64", "")
            if not vis_b64:
                raise SystemExit("Missing vis_jpg_b64 in result")
            vis_bytes = base64.b64decode(vis_b64)
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            out_img = out_dir / f"{stem}_vis.jpg"
            out_json = out_dir / f"{stem}_det.json"
            out_img.write_bytes(vis_bytes)
            out_json.write_text(json.dumps(result.get("detections", []), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved: {out_img}")
            print(f"Saved: {out_json}")
            return


if __name__ == "__main__":
    main()
