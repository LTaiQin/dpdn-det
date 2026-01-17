import argparse
import base64
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _now_s() -> float:
    return time.time()


class JobStore:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._load_existing()

    def _job_path(self, job_id: str) -> Path:
        return self.data_dir / f"{job_id}.json"

    def _load_existing(self):
        for p in self.data_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                job_id = data.get("job_id")
                if isinstance(job_id, str) and job_id:
                    self._jobs[job_id] = data
            except Exception:
                continue

    def _persist(self, job_id: str):
        p = self._job_path(job_id)
        p.write_text(json.dumps(self._jobs[job_id], ensure_ascii=False, indent=2), encoding="utf-8")

    def create_job(self, *, filename: str, image_b64: str, score_thresh: Optional[float]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",  # queued | processing | done | failed
                "created_at": _now_s(),
                "updated_at": _now_s(),
                "filename": filename,
                "score_thresh": score_thresh,
                "image_b64": image_b64,
                "worker_id": None,
                "error": None,
                "result": None,
            }
            self._persist(job_id)
        return job_id

    def get_job_public(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "filename": job.get("filename"),
                "score_thresh": job.get("score_thresh"),
                "error": job.get("error"),
                "result": job.get("result"),
            }

    def reserve_next(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            candidates = [j for j in self._jobs.values() if j.get("status") == "queued"]
            candidates.sort(key=lambda x: x.get("created_at", 0))
            if not candidates:
                return None
            job = candidates[0]
            job["status"] = "processing"
            job["worker_id"] = worker_id
            job["updated_at"] = _now_s()
            self._persist(job["job_id"])
            return {
                "job_id": job["job_id"],
                "filename": job.get("filename"),
                "score_thresh": job.get("score_thresh"),
                "image_b64": job.get("image_b64"),
            }

    def complete(self, job_id: str, result: Dict[str, Any], worker_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get("worker_id") != worker_id:
                return False
            job["status"] = "done"
            job["result"] = result
            job["error"] = None
            job["updated_at"] = _now_s()
            self._persist(job_id)
            return True

    def fail(self, job_id: str, error: str, worker_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.get("worker_id") != worker_id:
                return False
            job["status"] = "failed"
            job["error"] = error
            job["updated_at"] = _now_s()
            self._persist(job_id)
            return True


class RelayApp:
    def __init__(self, *, token: str, store: JobStore, max_image_bytes: int):
        self.token = token
        self.store = store
        self.max_image_bytes = int(max_image_bytes)

    def _check_token(self, headers) -> Tuple[bool, str]:
        got = headers.get("X-Relay-Token", "")
        if not self.token:
            return True, ""
        if got != self.token:
            return False, "Unauthorized"
        return True, ""

    def create_job_from_request(self, req: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        image_b64 = req.get("image_b64")
        if not isinstance(image_b64, str) or not image_b64:
            return None, "Missing image_b64"
        filename = req.get("filename") or "image"
        score_thresh = req.get("score_thresh", None)
        if score_thresh is not None:
            try:
                score_thresh = float(score_thresh)
            except Exception:
                return None, "Invalid score_thresh"
        try:
            raw = base64.b64decode(image_b64, validate=True)
        except Exception:
            return None, "Invalid base64 image"
        if len(raw) > self.max_image_bytes:
            return None, f"Image too large: {len(raw)} bytes > {self.max_image_bytes}"
        job_id = self.store.create_job(filename=str(filename), image_b64=image_b64, score_thresh=score_thresh)
        return job_id, None


def make_handler(app: RelayApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, payload: Dict[str, Any]):
            body = _json_bytes(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            ok, err = app._check_token(self.headers)
            if not ok and self.path.startswith("/api/"):
                self._send(401, {"ok": False, "error": err})
                return

            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True})
                return

            if self.path.startswith("/api/v1/jobs/"):
                parts = self.path.split("/")
                if len(parts) >= 5:
                    job_id = parts[4]
                    job = app.store.get_job_public(job_id)
                    if job is None:
                        self._send(404, {"ok": False, "error": "Not found"})
                        return
                    self._send(200, {"ok": True, "job": job})
                    return

            self._send(404, {"ok": False, "error": "Not found"})

        def do_POST(self):
            ok, err = app._check_token(self.headers)
            if not ok:
                self._send(401, {"ok": False, "error": err})
                return

            if self.path.rstrip("/") == "/api/v1/jobs":
                try:
                    req = self._read_json()
                    job_id, error = app.create_job_from_request(req)
                    if error:
                        self._send(400, {"ok": False, "error": error})
                        return
                    self._send(200, {"ok": True, "job_id": job_id})
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            if self.path.rstrip("/") == "/api/v1/jobs/next":
                try:
                    req = self._read_json()
                    worker_id = req.get("worker_id") or ""
                    if not isinstance(worker_id, str) or not worker_id:
                        self._send(400, {"ok": False, "error": "Missing worker_id"})
                        return
                    job = app.store.reserve_next(worker_id=worker_id)
                    self._send(200, {"ok": True, "job": job})
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            if self.path.startswith("/api/v1/jobs/") and self.path.endswith("/result"):
                try:
                    parts = self.path.split("/")
                    job_id = parts[4] if len(parts) >= 6 else ""
                    req = self._read_json()
                    worker_id = req.get("worker_id") or ""
                    result = req.get("result")
                    if not job_id:
                        self._send(400, {"ok": False, "error": "Missing job_id"})
                        return
                    if not isinstance(worker_id, str) or not worker_id:
                        self._send(400, {"ok": False, "error": "Missing worker_id"})
                        return
                    if not isinstance(result, dict):
                        self._send(400, {"ok": False, "error": "Missing result"})
                        return
                    if not app.store.complete(job_id=job_id, result=result, worker_id=worker_id):
                        self._send(409, {"ok": False, "error": "Job not reserved by this worker or not found"})
                        return
                    self._send(200, {"ok": True})
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            if self.path.startswith("/api/v1/jobs/") and self.path.endswith("/fail"):
                try:
                    parts = self.path.split("/")
                    job_id = parts[4] if len(parts) >= 6 else ""
                    req = self._read_json()
                    worker_id = req.get("worker_id") or ""
                    error = req.get("error") or ""
                    if not job_id:
                        self._send(400, {"ok": False, "error": "Missing job_id"})
                        return
                    if not isinstance(worker_id, str) or not worker_id:
                        self._send(400, {"ok": False, "error": "Missing worker_id"})
                        return
                    if not isinstance(error, str) or not error:
                        self._send(400, {"ok": False, "error": "Missing error"})
                        return
                    if not app.store.fail(job_id=job_id, error=error, worker_id=worker_id):
                        self._send(409, {"ok": False, "error": "Job not reserved by this worker or not found"})
                        return
                    self._send(200, {"ok": True})
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                return

            self._send(404, {"ok": False, "error": "Not found"})

    return Handler


def parse_args():
    p = argparse.ArgumentParser(description="Public relay server (VPS) for cross-network inference.")
    p.add_argument("--host", default="0.0.0.0", help="Bind host on VPS.")
    p.add_argument("--port", type=int, default=18081, help="Bind port on VPS (open this in firewall).")
    p.add_argument("--data-dir", default="./relay_data", help="Directory to store jobs.")
    p.add_argument("--token", default=os.environ.get("RELAY_TOKEN", ""), help="Shared token (X-Relay-Token).")
    p.add_argument("--max-image-bytes", type=int, default=10 * 1024 * 1024, help="Max upload size.")
    return p.parse_args()


def main():
    args = parse_args()
    store = JobStore(data_dir=args.data_dir)
    app = RelayApp(token=args.token, store=store, max_image_bytes=args.max_image_bytes)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"[relay] listening on http://{args.host}:{args.port}")
    print("[relay] health: GET  /health")
    print("[relay] submit: POST /api/v1/jobs (client)")
    print("[relay] poll:   GET  /api/v1/jobs/<job_id> (client)")
    print("[relay] next:   POST /api/v1/jobs/next (worker)")
    print("[relay] done:   POST /api/v1/jobs/<job_id>/result (worker)")
    server.serve_forever()


if __name__ == "__main__":
    main()

