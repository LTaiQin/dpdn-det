import argparse
import base64
import json
import socket
import time
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict, token: str, timeout_s: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json", "X-Relay-Token": token})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def infer_local(infer_url: str, image_b64: str, score_thresh):
    payload = {"image_b64": image_b64}
    if score_thresh is not None:
        payload["score_thresh"] = float(score_thresh)
    data = json.dumps(payload).encode("utf-8")
    req = Request(infer_url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if not out.get("ok", False):
        raise RuntimeError(out.get("error", "infer_server error"))
    return out["result"]


def parse_args():
    p = argparse.ArgumentParser(description="Worker poller (run near GPU server inside aTrust).")
    p.add_argument("--relay", required=True, help="Relay base url, e.g. http://47.237.1.93:18081")
    p.add_argument("--token", required=True, help="Relay token (X-Relay-Token).")
    p.add_argument("--infer-url", default="http://127.0.0.1:18080/infer", help="Local infer_server endpoint.")
    p.add_argument("--worker-id", default="", help="Worker ID (default: hostname).")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between polls when idle.")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token.strip().strip("'\"")
    worker_id = args.worker_id or socket.gethostname()
    relay = args.relay.rstrip("/")

    while True:
        try:
            resp = post_json(relay + "/api/v1/jobs/next", {"worker_id": worker_id}, token=token, timeout_s=30)
            if not resp.get("ok", False):
                raise RuntimeError(resp.get("error", "relay error"))
            job = resp.get("job")
            if not job:
                time.sleep(args.poll_interval)
                continue

            job_id = job["job_id"]
            score_thresh = job.get("score_thresh", None)
            image_b64 = job["image_b64"]

            try:
                base64.b64decode(image_b64, validate=True)
                result = infer_local(args.infer_url, image_b64=image_b64, score_thresh=score_thresh)
                post_json(
                    relay + f"/api/v1/jobs/{job_id}/result",
                    {"worker_id": worker_id, "result": result},
                    token=token,
                    timeout_s=60,
                )
            except Exception as exc:
                post_json(
                    relay + f"/api/v1/jobs/{job_id}/fail",
                    {"worker_id": worker_id, "error": str(exc)},
                    token=token,
                    timeout_s=60,
                )
        except Exception:
            time.sleep(max(args.poll_interval, 2.0))


if __name__ == "__main__":
    main()
