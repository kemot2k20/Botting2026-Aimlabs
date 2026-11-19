from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Target:
    x: float
    y: float
    dx: float
    dy: float
    area: int


class TargetDetector:
    def __init__(
            self,
            hsv_lower: tuple[int, int, int] = (80, 120, 120),
            hsv_upper: tuple[int, int, int] = (105, 255, 255),
            min_area: int = 80,
    ) -> None:
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

    def set_hsv_range(
            self,
            hsv_lower: tuple[int, int, int],
            hsv_upper: tuple[int, int, int],
    ) -> None:
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

    def detect(self, frame: np.ndarray) -> tuple[Target | None, np.ndarray]:
        bgr = frame[:, :, :3] if frame.shape[2] == 4 else frame

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        if num_labels <= 1:
            return None, mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_mask = areas >= self.min_area

        if not np.any(valid_mask):
            return None, mask

        valid_centroids = centroids[1:][valid_mask]
        valid_areas = areas[valid_mask]

        height, width = bgr.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0

        deltas = valid_centroids - np.array([center_x, center_y])

        dist_sq = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
        best_idx = int(np.argmin(dist_sq))

        best_x, best_y = valid_centroids[best_idx]
        best_dx, best_dy = deltas[best_idx]

        target = Target(
            x=float(best_x),
            y=float(best_y),
            dx=float(best_dx),
            dy=float(best_dy),
            area=int(valid_areas[best_idx]),
        )
        return target, mask