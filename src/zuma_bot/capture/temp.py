import cv2
import numpy as np
import math

# color ranges — start here and tune with sample frames (H, S, V)
COLOR_RANGES = {
    "blue":   ((90,  80,  60), (130, 255, 255)),
    "green":  ((50,  80,  60), (85,  255, 255)),
    "yellow": ((18, 100, 100), (35, 255, 255)),
    "purple": ((135, 60,  50), (165, 255, 255)),
}

def split_component_and_get_circles(component_mask, gray_roi,
                                    peak_thresh_rel=0.4, min_seed_area=20):
    """
    Given a binary component_mask (uint8 0/255) and the grayscale roi,
    split touching objects with distance transform + watershed and return
    list of (cx, cy, r) in ROI coordinates.
    """
    # distance transform
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    # find peaks (sure foreground)
    ret, sure_fg = cv2.threshold(dist, peak_thresh_rel * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # remove tiny seeds
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    good_seeds = np.zeros_like(sure_fg)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_seed_area:
            good_seeds[labels == lbl] = 255
    if cv2.countNonZero(good_seeds) == 0:
        # fallback: treat whole mask as single object
        cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return []
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        return [(int(cx), int(cy), int(r))]

    # prepare markers for watershed
    num_markers, markers = cv2.connectedComponents(good_seeds)
    markers = markers + 1  # so background != 0
    unknown = cv2.bitwise_and(cv2.bitwise_not(good_seeds), component_mask)
    markers[unknown == 255] = 0

    # need a 3-channel image for watershed; use blurred roi to preserve edges
    if len(gray_roi.shape) == 2:
        img_color = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    else:
        img_color = gray_roi.copy()

    # watershed (markers mutated in place)
    markers = cv2.watershed(img_color, markers.astype(np.int32))

    out_circles = []
    max_label = markers.max()
    for lab in range(2, max_label + 1):  # labels 2..N are objects
        comp = np.uint8(markers == lab) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        out_circles.append((int(cx), int(cy), int(max(1, int(r)))))

    return out_circles

def get_color_label_from_masked_hsv(hsv, contour):
    """Return the best matching color name for pixels inside contour."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    if cv2.countNonZero(mask) == 0:
        return None
    pixels = hsv[mask == 255]
    # compute circular mean for hue
    hues = pixels[:, 0].astype(float) * (np.pi / 90.0)  # scale
    mean_h = np.arctan2(np.mean(np.sin(hues)), np.mean(np.cos(hues)))
    mean_h_deg = (mean_h * (90.0 / np.pi)) % 180
    mean_sat = np.mean(pixels[:, 1])
    mean_val = np.mean(pixels[:, 2])

    # check each range by simple membership test on the median pixel
    med = np.median(pixels, axis=0).astype(int)
    for name, (low, high) in COLOR_RANGES.items():
        if (low[0] <= med[0] <= high[0]) and (low[1] <= med[1] <= high[1]) and (low[2] <= med[2] <= high[2]):
            return name
    # fallback: choose nearest hue by difference (simple)
    hue = med[0]
    best = None
    best_dist = 1e9
    for name, (low, high) in COLOR_RANGES.items():
        # treat hue circularly
        center = (low[0] + high[0]) / 2.0
        dist = min(abs(center - hue), 180 - abs(center - hue))
        if dist < best_dist:
            best_dist = dist
            best = name
    return best

def detect_balls_from_bgr(img_bgr, visualize=True, min_area_ratio=0.01, max_area_ratio=0.1, use_hough_fallback=True):
    """
    Detect circular balls in the given BGR image (in-memory).
    Returns (vis_img, detections) where detections is a list of dicts:
      {'center': (x,y), 'radius': r, 'color': 'blue'|'green'|'...'}
    """
    if img_bgr is None:
        return None, []

    H, W = img_bgr.shape[:2]
    img_area = float(H * W)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # gentle blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    vis = img_bgr.copy() if visualize else None
    detections = []

    # Create combined mask (we will iterate colors separately)
    for color_name, (low, high) in COLOR_RANGES.items():
        low = np.array(low, dtype=np.uint8)
        high = np.array(high, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

        # morphological clean
        k = max(1, int(round(min(W, H) * 0.002)))  # kernel size proportional to image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            # relative-area filter
            if not (min_area_ratio * img_area <= area <= max_area_ratio * img_area):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            # keep fairly circular shapes
            if circularity < 0.5:
                # not circular enough; optionally continue to next contour
                # but still allow Hough fallback below
                pass

            # estimate circle using minEnclosingCircle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(round(cx)), int(round(cy)), int(round(radius))
            if radius <= 2:
                continue

            # optionally refine ROI and run Hough there if desired / needed
            detection_added = False
            if use_hough_fallback:
                # build small ROI padded
                pad = max(2, int(radius * 0.25))
                x1 = max(0, cx - radius - pad); y1 = max(0, cy - radius - pad)
                x2 = min(W, cx + radius + pad); y2 = min(H, cy + radius + pad)
                roi_gray = blurred[y1:y2, x1:x2]
                if roi_gray.size == 0:
                    continue
                # smaller image speeds Hough; scale down if ROI large
                scale = 1.0
                max_side_for_hough = 160
                if max(roi_gray.shape[:2]) > max_side_for_hough:
                    scale = max_side_for_hough / float(max(roi_gray.shape[:2]))
                    roi_small = cv2.resize(roi_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    roi_small = roi_gray

                # tune param2: higher => fewer false positives
                dp = 1.2
                minDist = max(10, int(radius * scale * 0.5))
                minR = max(2, int(radius * scale * 0.6))
                maxR = max(3, int(radius * scale * 1.6))
                circles = cv2.HoughCircles(roi_small, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                           param1=100, param2=30, minRadius=minR, maxRadius=maxR)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    # take the best (first) circle
                    c = circles[0, 0]
                    ccx = int(round(c[0] / scale)) + x1
                    ccy = int(round(c[1] / scale)) + y1
                    rr  = int(round(c[2] / scale))
                    color_label = color_name  # we iterated by color, so use it
                    detections.append({'center': (ccx, ccy), 'radius': rr, 'color': color_label})
                    detection_added = True

            if not detection_added:
                # fallback to contour-based detection
                color_label = get_color_label_from_masked_hsv(hsv, cnt) or color_name
                detections.append({'center': (cx, cy), 'radius': radius, 'color': color_label})

            # draw
            if visualize:
                cv2.circle(vis, (detections[-1]['center'][0], detections[-1]['center'][1]),
                           int(detections[-1]['radius']), (0, 255, 0), 2)
                cv2.circle(vis, (detections[-1]['center'][0], detections[-1]['center'][1]), 2, (0, 0, 255), 2)
                cv2.putText(vis, color_label, (detections[-1]['center'][0] - 10, detections[-1]['center'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # remove duplicates (centers very close) by simple NMS
    final = []
    used = [False] * len(detections)
    for i, di in enumerate(detections):
        if used[i]:
            continue
        cx, cy = di['center']
        r = di['radius']
        group = [i]
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            cx2, cy2 = detections[j]['center']
            if (cx - cx2) ** 2 + (cy - cy2) ** 2 < (max(r, detections[j]['radius']) * 0.6) ** 2:
                group.append(j)
                used[j] = True
        # pick largest radius / or average
        if len(group) == 1:
            final.append(di)
        else:
            # choose detection with largest radius
            pick = max(group, key=lambda idx: detections[idx]['radius'])
            final.append(detections[pick])
    return vis, final
