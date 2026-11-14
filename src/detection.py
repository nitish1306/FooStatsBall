"""
Detection and tracking pipeline using sports-tuned YOLO11 + BoT-SORT on GPU.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

CUSTOM_BOTSORT_CONFIG = str(Path(__file__).parent.parent / "sports_botsort.yaml")


class DetectionTracker:
    """YOLO11 sports-tuned detection + BoT-SORT tracking pipeline."""

    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cuda", imgsz: int = 1280):
        """
        Args:
            model_path: Path to YOLO model weights.
            device: 'cuda' for GPU, 'cpu' for CPU.
            imgsz: Higher resolution inference (1280x720/1280 for stabilizing far touchline players).
        """
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.tracker_cfg = CUSTOM_BOTSORT_CONFIG

        # COCO class IDs: 0 = person, 32 = sports ball
        self.person_class_id = 0
        self.ball_class_id = 32

    def detect_and_track(self, frame: np.ndarray, conf: float = 0.25):
        """
        Run detection and tracking on a single frame with imgsz=1280 and sports-tuned tracker.
        Returns a tuple of (players, ball_center).
        """
        results = self.model.track(
            source=frame,
            conf=conf,
            device=self.device,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
            imgsz=self.imgsz,
            classes=[self.person_class_id, self.ball_class_id]
        )[0]

        players = []
        ball_center = None
        best_ball_conf = -1.0

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(boxes)

            for i in range(len(boxes)):
                cls_id = classes[i]
                b = xyxy[i]
                c = float(confs[i])
                tid = int(track_ids[i]) if i < len(track_ids) else -1

                if cls_id == self.person_class_id:
                    # Ground contact point: center bottom of the bbox
                    foot_x = float((b[0] + b[2]) / 2.0)
                    foot_y = float(b[3])
                    players.append({
                        "tracker_id": tid,
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "foot_pixel": [foot_x, foot_y],
                        "confidence": c
                    })
                elif cls_id == self.ball_class_id:
                    if c > best_ball_conf:
                        best_ball_conf = c
                        bx = float((b[0] + b[2]) / 2.0)
                        by = float((b[1] + b[3]) / 2.0)
                        ball_center = [bx, by]

        return players, ball_center

    def reset_tracker(self):
        """Reinitialize model to reset persistent tracker states."""
        self.model = YOLO(self.model.model_name or "yolo11n.pt")
