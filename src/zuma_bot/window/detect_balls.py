import cv2 as cv
import numpy as np
import math

# Derived from your measurements (radius ≈ 26–30 px at full scale)
RADIUS_REL_MIN = 0.040 / 2    # ≈ 0.023
RADIUS_REL_MAX = 0.060 / 2    # ≈ 0.030
# ----------------------------
COLOR_PROTOTYPES = {
    "blue":   110,
    "yellow":  25,
    "green":   60,
    "purple": 145,
}

# Classification thresholds (tweak if necessary)
HUE_WINDOW = 15           # degrees each side for pixel-level color membership
SAT_THRESH = 60           # minimum average saturation to consider (0..255)
VAL_THRESH = 60           # minimum average value to consider (0..255)
PROPORTION_THRESH = 0.45  # fraction of circle pixels that must match the chosen color
MIN_PIXELS = 40           # minimal pixels inside circle to consider valid

def circular_hue_diff(h1, h2):
    d = abs(h1 - h2)
    return np.minimum(d, 180 - d)

def detect_zuma_balls(img):
    h, w = img.shape[:2]
    ref = min(w, h)

    min_radius = int(RADIUS_REL_MIN * ref)
    max_radius = int(RADIUS_REL_MAX * ref)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.15,
        minDist=min_radius * 1.6,
        param1=120,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        print("No balls detected.")
        return np.empty((0, 3), dtype=int)

    circles = np.uint16(np.around(circles[0]))
    return circles

def classify_circles(img, circles,
                     prototypes=COLOR_PROTOTYPES,
                     hue_window=HUE_WINDOW,
                     sat_thresh=SAT_THRESH,
                     val_thresh=VAL_THRESH,
                     proportion_thresh=PROPORTION_THRESH,
                     min_pixels=MIN_PIXELS):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    results = []
    for (x, y, r) in circles:
        # create circular mask slightly smaller than r to avoid edge artifacts
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        inner_r = max(1, r - 2)
        cv.circle(mask, (int(x), int(y)), inner_r, 255, -1)

        pixels = hsv[mask == 255]
        if pixels.size == 0 or len(pixels) < min_pixels:
            continue

        # pixels: array of [H,S,V]
        h_pixels = pixels[:, 0].astype(np.int16)
        s_pixels = pixels[:, 1].astype(np.int16)
        v_pixels = pixels[:, 2].astype(np.int16)

        avg_s = int(np.mean(s_pixels))
        avg_v = int(np.mean(v_pixels))
        if avg_s < sat_thresh or avg_v < val_thresh:
            # too desaturated / too dark -> likely false detection (edge, UI element, etc.)
            continue

        # compute average hue (careful with circular nature)
        # convert to complex vectors to average angle
        angles = (h_pixels.astype(np.float64) / 180.0) * 2 * np.pi
        vx = np.cos(angles).mean()
        vy = np.sin(angles).mean()
        avg_h_rad = math.atan2(vy, vx)
        if avg_h_rad < 0:
            avg_h_rad += 2 * math.pi
        avg_h = (avg_h_rad / (2 * math.pi)) * 180.0
        avg_h = float(avg_h)  # in 0..180

        # choose prototype with minimum circular hue distance
        best_color = None
        best_dist = 1e9
        for cname, ch in prototypes.items():
            d = min(abs(avg_h - ch), 180 - abs(avg_h - ch))
            if d < best_dist:
                best_dist = d
                best_color = cname

        # verify that a sufficient proportion of pixels actually match the chosen color
        ch_val = prototypes[best_color]
        hue_diffs = circular_hue_diff(h_pixels, ch_val)
        matched_mask = (hue_diffs <= hue_window) & (s_pixels >= sat_thresh) & (v_pixels >= val_thresh)
        proportion = matched_mask.sum() / len(h_pixels)

        if proportion < proportion_thresh:
            # not enough pixels match; likely false detection or mixed/noisy region
            continue

        # valid ball -> append record
        results.append({
            "x": int(x),
            "y": int(y),
            "radius": int(r),
            "color": best_color,
        })

    return results

def annotate_and_show(img, balls):
    out = img.copy()
    for b in balls:
        x, y, r = b["x"], b["y"], b["radius"]
        color_name = b["color"]
        # choose BGR text color for readability
        label_color = (0, 255, 0)
        cv.circle(out, (x, y), r, label_color, 2)
        cv.circle(out, (x, y), 2, (0, 0, 255), 3)
        cv.putText(out, f"{color_name}", (x - r, y - r - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv.LINE_AA)
    cv.imshow("Detected & Classified Balls", out)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def start_ball_detection(img):
    if img is None:
        raise FileNotFoundError("Image not found: ")

    circles = detect_zuma_balls(img)
    print(f"Raw Hough detections: {len(circles)}")
    
    balls = classify_circles(img, circles)
    print(f"Validated classified balls: {len(balls)}")
    
    simplified = [{"x": b["x"], "y": b["y"], "radius": b["radius"], "color": b["color"]} for b in balls]
    for b in simplified:
        print(b)

    annotate_and_show(img, balls)
    return simplified