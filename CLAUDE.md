# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (includes dev extras)
pip install -e ".[dev]"

# Run the server
python -m server.app -c config/default.yaml -s config/shelf_layout.yaml

# API-only mode (no pipeline, useful for debugging routes)
python -m server.app --no-pipeline

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_event_engine.py

# Export YOLO model to ONNX (development machine)
python tools/model_export.py -m models/yolo11n.pt -o models/yolo11n.onnx

# Convert ONNX to RKNN for RK3588 deployment (development machine, requires rknn-toolkit2)
python tools/model_export_rknn.py -i models/yolo11n.onnx -o models/yolo11n.rknn
# With INT8 quantization (~2x faster inference, needs calibration images)
python tools/model_export_rknn.py -i models/yolo11n.onnx -o models/yolo11n_int8.rknn --int8 --calib-dir data/raw/calib/
```

## Architecture

This is a real-time shelf monitoring system that detects **pick** (item removed) and **place** (item returned) events from camera feeds using YOLO + ByteTrack.

### Pipeline data flow (per frame)

```
VideoSource → ROI Crop → YOLODetector → ByteTracker → ShelfState → EventEngine → EventOutput → WebSocket/REST
```

1. **`pipeline/io/source.py`** — `VideoSource` abstract base; concrete impls for RTSP, USB, and file. Reads frames in a background thread into a queue. Controlled via `source.start()` / `source.stop()`.

2. **`pipeline/detector/yolo_detector.py`** — `YOLODetector` wraps ONNX Runtime inference. Call `detector.load()` then `detector.detect(frame)` → `ndarray` of shape `(N, 6)`: `[x1, y1, x2, y2, class_id, score]`. Input images are preprocessed to `640×640` internally. For RK3588 edge deployment, `pipeline/detector/rknn_detector.py` provides `RKNNDetector` with an identical interface backed by the NPU via `rknnlite`; switch by setting `detector.backend: rknn` in `config/default.yaml`.

3. **`pipeline/tracker/byte_tracker.py`** — `ByteTracker` wraps the ByteTrack algorithm. `tracker.update(detections)` → `ndarray (M, 7)`: `[x1, y1, x2, y2, track_id, class_id, score]`. **Not enabled in the current MVP** — the pipeline runs without it by passing detections directly to ShelfState.

4. **`pipeline/shelf_state.py`** — `ShelfState` is the central state layer. It maps detection boxes to fixed shelf slots via IoU matching, then runs a per-slot debounce (N consecutive frames of the same state before confirming). `state.update(tracks, timestamp)` → `dict[slot_id, bool]` (confirmed occupancy). `state.get_snapshot()` returns the full slot status for the REST API.

5. **`pipeline/event_engine.py`** — `EventEngine` does a simple frame-diff on the debounced state dict. `True→False` = pick, `False→True` = place. Debounce is already done by ShelfState, so EventEngine fires on every state change it sees. Fires an optional `on_event` callback and accumulates history in `_events`.

6. **`pipeline/output.py`** — `EventOutput` handles snapshot saving (JPEG to `data/snapshots/`) and SQLite logging. `broadcast_async(event, ws_clients)` fans out to all connected WebSocket clients.

7. **`server/app.py`** — FastAPI app. The `main()` function loads config, calls `init_pipeline()`, then starts `run_pipeline_loop()` as an asyncio task before handing off to uvicorn. Module-level globals hold component singletons (`video_source`, `detector`, `tracker`, etc.).

### Slot-based state model

Shelves are divided into fixed **slots** defined in `config/shelf_layout.yaml`. Each slot has a normalized `roi: [x1, y1, x2, y2]` and an expected `sku_id`. ShelfState matches detection boxes to slots by IoU; a slot is `occupied` when a detection's IoU with its RoI exceeds `min_iou_for_slot` (default 0.3) for `debounce_frames` (default 5) consecutive frames. This is the core design choice — tracking fixed regions instead of individual objects avoids re-ID complexity and makes the system robust to partial occlusion.

### Configuration

Two YAML files drive the system:
- **`config/default.yaml`** — pipeline params (input source, detector thresholds, tracker settings, debounce, server port, output paths)
- **`config/shelf_layout.yaml`** — shelf geometry: `shelf_id`, `camera_id`, frame `width`/`height`, and the `slots` list

Class-to-SKU mapping lives in **`config/model_mapping.yaml`**.

Models must be in ONNX format under `models/`. The `tools/model_export.py` script converts from PyTorch (ultralytics) to ONNX.

### Key constraints (V1)

- One item per slot, single layer — binary occupancy only, no counting.
- ByteTracker is wired but **disabled by default**; ShelfState works without track IDs.
- Occlusion gating (freezing a slot's state machine when a person overlaps its RoI) is specified in the design doc (`files/design.md`) but not yet implemented in code.
