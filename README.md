# Self-Driving Vision Module (Demo)

A minimal, self-contained full‑stack demo replicating the portfolio project description: real‑time lane detection, object detection (YOLO via OpenCV DNN), and simple collision risk prediction.

## Structure

```
backend/
  app.py            # Flask app with /process-image
  model/            # Place YOLO files here (see below)
frontend/
  index.html        # UI: upload image, draw results
  app.js            # Client-side logic and canvas drawing
requirements.txt
```

## Setup

1) Create and activate a Python 3.8+ environment.

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Download YOLOv3-tiny files and place them under `backend/model/`:

- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`
- `coco.names`

You can obtain them from the official sources:

- Config: `https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg`
- Weights: `https://pjreddie.com/media/files/yolov3-tiny.weights`
- COCO names: `https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`

If these files are not present, the backend will still run but object detection will return an empty list.

## Run

In one terminal (backend):

```
python backend/app.py
```

In a static file server (frontend), open `frontend/index.html` in a browser. You can also open the file directly.

The default endpoint is `http://localhost:5000/process-image`. Change it in the page if your backend runs elsewhere.

## API

POST `/process-image`

Form field: `image` (file)

Response JSON:

```
{
  "lanes": [{"x1": int, "y1": int, "x2": int, "y2": int}, ...],
  "objects": [
    {"bbox": [x, y, w, h], "label": str, "confidence": float, "area_ratio": float, "high_risk": bool},
    ...
  ],
  "yolo_model_loaded": bool
}
```

Collision risk is flagged when a detection's bounding box area is ≥ 30% of the image area.

## Notes

- Lane detection uses Gaussian blur, Canny edges, ROI masking, and probabilistic Hough transform.
- Object detection uses OpenCV DNN with YOLOv3-tiny; NMS applied.
- This is a demo and not optimized for production or real-time constraints.

# GitHub Pages Site
