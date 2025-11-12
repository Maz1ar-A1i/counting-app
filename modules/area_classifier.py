import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class AreaClassifier:
    """
    Simple area-wise classifier using background subtraction and contour area thresholds.
    Designed to be independent from ML models and the existing detector.
    """

    def __init__(self, min_area: int = 500, area_bins: Optional[Dict[str, Tuple[int, int]]] = None):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = int(min_area)
        # area_bins: dict name -> (min, max)
        self.area_bins: Dict[str, Tuple[int, int]] = area_bins or {
            "small": (0, 1500),
            "medium": (1500, 5000),
            "large": (5000, 1_000_000)
        }
        self.fps_start = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def update_settings(self, min_area: Optional[int] = None, area_bins: Optional[Dict[str, Tuple[int, int]]] = None):
        if min_area is not None:
            self.min_area = int(min_area)
        if area_bins is not None:
            self.area_bins = {k: tuple(v) for k, v in area_bins.items()}

    def process(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Process a frame and return classification results and an annotated frame.
        """
        mask = self.bg_subtractor.apply(frame)
        # Clean mask
        mask = cv2.medianBlur(mask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_counts = {k: 0 for k in self.area_bins.keys()}
        total_regions = 0
        annotations = []

        annotated = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            total_regions += 1

            size_label = None
            for name, (a_min, a_max) in self.area_bins.items():
                if a_min <= area < a_max:
                    size_label = name
                    break
            if size_label is None:
                size_label = "unclassified"
                if size_label not in class_counts:
                    class_counts[size_label] = 0
            class_counts[size_label] = class_counts.get(size_label, 0) + 1

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"{size_label} ({int(area)})",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
            annotations.append({"bbox": [x, y, x + w, y + h], "area": float(area), "label": size_label})

        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        cv2.putText(
            annotated,
            f"Area FPS: {self.fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        return {
            "counts": class_counts,
            "total_regions": total_regions,
            "annotations": annotations,
            "fps": self.fps,
            "mask_preview": None,  # could add if needed
            "frame": annotated
        }



