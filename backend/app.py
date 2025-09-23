import os
import io
from typing import List, Tuple, Dict, Any

from flask import Flask, request, jsonify
import numpy as np
import cv2


def create_app() -> Flask:
    app = Flask(__name__)

    # Model paths (users should place files under backend/model/)
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    yolo_cfg_path = os.path.join(model_dir, "yolov3-tiny.cfg")
    yolo_weights_path = os.path.join(model_dir, "yolov3-tiny.weights")
    yolo_names_path = os.path.join(model_dir, "coco.names")

    # Attempt to load YOLO if present; otherwise defer with None
    net = None
    output_layer_names: List[str] = []
    class_labels: List[str] = []
    if os.path.exists(yolo_cfg_path) and os.path.exists(yolo_weights_path):
        try:
            net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights_path)
            try:
                # If OpenCV DNN with CUDA is available, enable it; otherwise CPU
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass

            layer_names = net.getLayerNames()
            output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        except Exception:
            net = None
            output_layer_names = []

    if os.path.exists(yolo_names_path):
        with open(yolo_names_path, "r", encoding="utf-8") as f:
            class_labels = [line.strip() for line in f.readlines() if line.strip()]

    def lane_detection(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect lane line segments via Gaussian blur, Canny, ROI, Prob. Hough.

        Returns list of (x1, y1, x2, y2) segments.
        """
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        height, width = edges.shape[:2]
        # Define a trapezoidal ROI for typical front-facing dashcam
        mask = np.zeros_like(edges)
        polygon = np.array([
            [
                (int(0.1 * width), height),
                (int(0.45 * width), int(0.6 * height)),
                (int(0.55 * width), int(0.6 * height)),
                (int(0.9 * width), height),
            ]
        ], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(0.05 * width),
            maxLineGap=20,
        )

        segments: List[Tuple[int, int, int, int]] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segments.append((int(x1), int(y1), int(x2), int(y2)))
        return segments

    def object_detection(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLOv3-tiny with OpenCV DNN; return list of detections as dicts
        with bbox [x, y, w, h], label, confidence. Applies NMS.
        """
        height, width = image_bgr.shape[:2]
        if net is None or not output_layer_names:
            # Gracefully return empty when model not available
            return []

        blob = cv2.dnn.blobFromImage(image_bgr, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layer_names)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)

        results: List[Dict[str, Any]] = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                label = class_labels[class_ids[i]] if 0 <= class_ids[i] < len(class_labels) else str(class_ids[i])
                results.append(
                    {
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "label": label,
                        "confidence": float(confidences[i]),
                    }
                )
        return results

    def collision_prediction(image_shape: Tuple[int, int, int], detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flag objects as high_risk if bbox area exceeds 30% of frame area."""
        height, width = image_shape[:2]
        frame_area = float(height * width)
        annotated: List[Dict[str, Any]] = []
        for det in detections:
            x, y, w, h = det["bbox"]
            area = max(0, w) * max(0, h)
            rel = area / frame_area if frame_area > 0 else 0.0
            high_risk = rel >= 0.30
            annotated.append({**det, "area_ratio": rel, "high_risk": high_risk})
        return annotated

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/process-image", methods=["POST"])
    def process_image() -> Any:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided under form field 'image'"}), 400

        file_storage = request.files["image"]
        file_bytes = file_storage.read()
        if not file_bytes:
            return jsonify({"error": "Empty image payload"}), 400

        file_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"error": "Failed to decode image"}), 400

        lanes = lane_detection(image_bgr)
        detections = object_detection(image_bgr)
        risks = collision_prediction(image_bgr.shape, detections)

        response = {
            "lanes": [{"x1": x1, "y1": y1, "x2": x2, "y2": y2} for (x1, y1, x2, y2) in lanes],
            "objects": risks,
            "yolo_model_loaded": net is not None,
        }
        return jsonify(response)

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000, debug=True)

