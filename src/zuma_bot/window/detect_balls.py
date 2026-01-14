import cv2 as cv
import numpy as np
import math

# ----------------------------
# Ball size range (relative to min(image_w, image_h))
# (These were based on your earlier measurements)
RADIUS_REL_MIN = 0.040 / 2    # ~0.020
RADIUS_REL_MAX = 0.060 / 2    # ~0.030
# ----------------------------

COLOR_PROTOTYPES = {
    "blue":   110,
    "yellow":  30,
    "green":   70,
    "purple": 150,
}

# Classification thresholds
HUE_WINDOW = 15           # degrees each side
SAT_THRESH = 60
VAL_THRESH = 60
PROPORTION_THRESH = 0.45
MIN_PIXELS = 40

# Optional speed knob: downscale before Hough (1.0 = off)
DETECT_DOWNSCALE = 1.0    # try 0.75 or 0.6 if you want more speed


def circular_hue_diff(h_arr, h_scalar):
    """Circular distance on OpenCV hue wheel (0..179)."""
    h_scalar = int(h_scalar)
    d = np.abs(h_arr.astype(np.int16) - h_scalar)
    return np.minimum(d, 180 - d)


def detect_zuma_balls(img, downscale=DETECT_DOWNSCALE):
    """
    Detect circles with HoughCircles. Returns Nx3 array [x,y,r] in ORIGINAL scale.
    """
    h, w = img.shape[:2]

    if downscale != 1.0:
        new_w = max(1, int(w * downscale))
        new_h = max(1, int(h * downscale))
        small = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    else:
        small = img

    hs, ws = small.shape[:2]
    ref = min(ws, hs)

    min_radius = int(RADIUS_REL_MIN * ref)
    max_radius = int(RADIUS_REL_MAX * ref)

    if min_radius <= 0 or max_radius <= 0 or max_radius < min_radius:
        return np.empty((0, 3), dtype=np.int32)

    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(1, int(min_radius * 1.6)),
        param1=120,
        param2= 17,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return np.empty((0, 3), dtype=np.int32)

    circles = np.around(circles[0]).astype(np.float32)

    # Scale back up if downscaled
    if downscale != 1.0:
        circles[:, :3] /= float(downscale)

    circles = np.rint(circles).astype(np.int32)
    return circles


def classify_circles(
    img,
    circles,
    prototypes=COLOR_PROTOTYPES,
    hue_window=HUE_WINDOW,
    sat_thresh=SAT_THRESH,
    val_thresh=VAL_THRESH,
    proportion_thresh=PROPORTION_THRESH,
    min_pixels=MIN_PIXELS
):
    """
    Validate circles and classify their color.
    Returns list of dicts: {x,y,radius,color}
    """
    if circles is None or len(circles) == 0:
        return []

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]

    results = []
    for (x, y, r) in circles:
        x = int(x); y = int(y); r = int(r)
        inner_r = max(1, r - 2)

        # ROI bounding box around the circle (FAST: no full-image mask)
        x0 = max(0, x - inner_r)
        x1 = min(W, x + inner_r + 1)
        y0 = max(0, y - inner_r)
        y1 = min(H, y + inner_r + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = hsv[y0:y1, x0:x1]
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv.circle(mask, (x - x0, y - y0), inner_r, 255, -1)

        pixels = roi[mask == 255]
        if pixels.shape[0] < min_pixels:
            continue

        h_pixels = pixels[:, 0].astype(np.int16)
        s_pixels = pixels[:, 1].astype(np.int16)
        v_pixels = pixels[:, 2].astype(np.int16)

        avg_s = int(np.mean(s_pixels))
        avg_v = int(np.mean(v_pixels))
        if avg_s < sat_thresh or avg_v < val_thresh:
            continue

        # Average hue on a circle using unit vectors
        angles = (h_pixels.astype(np.float64) / 180.0) * 2.0 * np.pi
        vx = np.cos(angles).mean()
        vy = np.sin(angles).mean()
        avg_h_rad = math.atan2(vy, vx)
        if avg_h_rad < 0:
            avg_h_rad += 2 * math.pi
        avg_h = (avg_h_rad / (2 * math.pi)) * 180.0  # 0..180

        # Closest prototype by circular distance
        best_color = None
        best_dist = 1e9
        for cname, ch in prototypes.items():
            d = min(abs(avg_h - ch), 180 - abs(avg_h - ch))
            if d < best_dist:
                best_dist = d
                best_color = cname

        ch_val = prototypes[best_color]
        hue_diffs = circular_hue_diff(h_pixels, ch_val)

        matched = (hue_diffs <= hue_window) & (s_pixels >= sat_thresh) & (v_pixels >= val_thresh)
        proportion = float(np.count_nonzero(matched)) / float(len(h_pixels))

        if proportion < proportion_thresh:
            continue

        results.append({
            "x": x,
            "y": y,
            "radius": r,
            "color": best_color
        })

    return results


def draw_balls(img, balls):
    out = img.copy()
    for b in balls:
        x, y, r = b["x"], b["y"], b["radius"]
        label = b["color"]

        cv.circle(out, (x, y), r, (0, 255, 0), 2)
        cv.circle(out, (x, y), 2, (0, 0, 255), 3)
        cv.putText(out, label, (x - r, y - r - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    return out


def start_ball_detection(
    img,
    *,
    show=False,
    window_name="Detected & Classified Balls",
    debug=False,
    return_annotated=False,
    downscale=DETECT_DOWNSCALE
):
    """
    Main entry.
    Returns:
      - if return_annotated=False: list of balls
      - if return_annotated=True: (list_of_balls, annotated_image)
    """
    if img is None:
        raise FileNotFoundError("Image not found (img is None).")

    circles = detect_zuma_balls(img, downscale=downscale)
    balls = classify_circles(img, circles)

    if debug:
        print(f"Raw Hough detections: {len(circles)}")
        print(f"Validated classified balls: {len(balls)}")

    annotated = draw_balls(img, balls)

    if show:
        cv.imshow(window_name, annotated)

    simplified = [{"x": b["x"], "y": b["y"], "radius": b["radius"], "color": b["color"]} for b in balls]

    if return_annotated:
        return simplified, annotated
    return simplified
