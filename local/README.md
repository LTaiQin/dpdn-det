## Remote Single-Image Inference (Server + Windows Client)

## Scheme A (VPS Relay + GPU Worker Poller)

This is for cross-network use (and easy to extend to Android).

**Your VPS public IP**: `47.237.1.93`

### Ports

- VPS relay: `47.237.1.93:18081` (open this inbound on the VPS firewall/security group)
- GPU infer service (inside aTrust): `127.0.0.1:18080` (`second_14.1/tools/infer_server.py`)

### 1) VPS: run the relay server (runs on VPS)

Copy `second_14.1/tools/relay_server.py` to the VPS and run:

```bash
export RELAY_TOKEN='CHANGE_ME_LONG_RANDOM'
python relay_server.py --host 0.0.0.0 --port 18081 --data-dir ./relay_data --token "$RELAY_TOKEN"
```

Health check:
- `curl http://127.0.0.1:18081/health`
- From your local: `curl http://47.237.1.93:18081/health`

### 2) GPU server: run inference HTTP (runs on GPU server)

```bash
python tools/infer_server.py --host 127.0.0.1 --port 18080 \
  --config-file configs/coco/COCO_OVD_Food2K_PIS.yaml \
  --weights output/coco_ovd_food2k_PIS_with_cap_new/model_0079999.pth
```

### 3) GPU server: run worker poller (runs on GPU server)

Worker polls VPS for jobs and calls local `infer_server.py`:

```bash
python tools/worker_poller.py \
  --relay http://47.237.1.93:18081 \
  --token "$RELAY_TOKEN" \
  --infer-url http://127.0.0.1:18080/infer
```

### 4) Local PC (Windows): submit an image (runs on local PC)

```powershell
python local\submit_via_vps.py --vps http://47.237.1.93:18081 --token "CHANGE_ME_LONG_RANDOM" --pick
```

Note (Windows `cmd.exe`): use double-quotes or no quotes for `--token`. Single-quotes are treated as literal characters.

Outputs:
- `D:\pyCharmProjects\server\output\<name>_vis.jpg`
- `D:\pyCharmProjects\server\output\<name>_det.json`

### 1) On the server (Linux)

Start the inference service from `second_14.1/`:

```bash
python tools/infer_server.py \
  --host 127.0.0.1 \
  --port 18080 \
  --config-file configs/coco/COCO_OVD_Food2K_PIS.yaml \
  --weights output/coco_ovd_food2k_PIS_with_cap_new/model_0079999.pth
```

This binds **server** `127.0.0.1:18080` and keeps it private (recommended with SSH port-forward).

### 2) SSH port-forward (on Windows local machine)

Forward **local** `127.0.0.1:18080` -> **server** `127.0.0.1:18080`:

```powershell
ssh -L 18080:127.0.0.1:18080 <user>@<server_host>
```

Ports:
- Server listens on: `127.0.0.1:18080`
- Local forwards to: `127.0.0.1:18080`

### 3) On Windows (client)

Run:

```powershell
python local\client_infer.py --pick
```

Or specify an image:

```powershell
python local\client_infer.py --input C:\path\to\img.jpg
```

Outputs:
- Image: `D:\pyCharmProjects\server\output\<name>_vis.jpg`
- Detections JSON: `D:\pyCharmProjects\server\output\<name>_det.json`

### 4) Optional: PyQt GUI (Windows)

If you have `PyQt5` installed, you can use the richer GUI:

```powershell
python local\main.py
```

Features:
- Drag-and-drop images into a queue (batch inference)
- Server health check + copy SSH port-forward command
- Detections table (class/score/bbox) + quick open output folder
- Settings persistence (server URL / output dir / threshold)
