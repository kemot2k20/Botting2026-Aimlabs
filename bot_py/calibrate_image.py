import sys
import time
from pathlib import Path

import cv2
import numpy as np

from detector import TargetDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "samples" / "sample1.webp"
ROI_SIZE = 380
WIN_NAME = "HSV Calibrator"


def nothing(_: int) -> None:
    pass


def crop_center_roi(img: np.ndarray, roi_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    crop_w = min(w, roi_size)
    crop_h = min(h, roi_size)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    full_img = cv2.imread(str(image_path))
    if full_img is None:
        raise FileNotFoundError(f"Nie udało się wczytać obrazu: {image_path.resolve()}")

    roi_frame = crop_center_roi(full_img, ROI_SIZE)
    roi_h, roi_w = roi_frame.shape[:2]
    cx, cy = roi_w // 2, roi_h // 2

    detector = TargetDetector(min_area=50)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, roi_w * 2, roi_h + 260)
    cv2.moveWindow(WIN_NAME, 100, 100)

    cv2.createTrackbar("H_min", WIN_NAME, 80, 179, nothing)
    cv2.createTrackbar("H_max", WIN_NAME, 105, 179, nothing)
    cv2.createTrackbar("S_min", WIN_NAME, 120, 255, nothing)
    cv2.createTrackbar("S_max", WIN_NAME, 255, 255, nothing)
    cv2.createTrackbar("V_min", WIN_NAME, 120, 255, nothing)
    cv2.createTrackbar("V_max", WIN_NAME, 255, 255, nothing)
    cv2.createTrackbar("Min_Area", WIN_NAME, 50, 1000, nothing)

    while True:
        h_min = cv2.getTrackbarPos("H_min", WIN_NAME)
        h_max = cv2.getTrackbarPos("H_max", WIN_NAME)
        s_min = cv2.getTrackbarPos("S_min", WIN_NAME)
        s_max = cv2.getTrackbarPos("S_max", WIN_NAME)
        v_min = cv2.getTrackbarPos("V_min", WIN_NAME)
        v_max = cv2.getTrackbarPos("V_max", WIN_NAME)
        detector.min_area = cv2.getTrackbarPos("Min_Area", WIN_NAME)
        detector.set_hsv_range((h_min, s_min, v_min), (h_max, s_max, v_max))

        t0 = time.perf_counter()
        target, mask = detector.detect(roi_frame)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        debug_view = roi_frame.copy()
        cv2.drawMarker(debug_view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)

        if target is not None:
            tx, ty = int(target.x), int(target.y)
            cv2.circle(debug_view, (tx, ty), 8, (0, 0, 255), 2)
            cv2.line(debug_view, (cx, cy), (tx, ty), (0, 255, 255), 2)
            info = f"dx:{target.dx:+.0f} dy:{target.dy:+.0f} area:{target.area}"
        else:
            info = "Brak celu"

        cv2.putText(
            debug_view,
            f"{detect_ms:.2f} ms | {info}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # Sklejenie podglądu i maski w jeden obraz
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([debug_view, mask_bgr])
        cv2.imshow(WIN_NAME, combined)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("\n--- Skalibrowane wartości ---")
            print(f"hsv_lower = ({h_min}, {s_min}, {v_min})")
            print(f"hsv_upper = ({h_max}, {s_max}, {v_max})")
            print(f"min_area  = {detector.min_area}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()