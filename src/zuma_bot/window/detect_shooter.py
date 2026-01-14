import math
import numpy as np
import cv2 as cv

# How far from the board center we accept "frog balls"
# Tune this if needed (0.18..0.30 are typical)
SHOOTER_MAX_DIST_REL = 0.24

# If we fail to find 2 balls from global detections, we do a fallback local detect
FALLBACK_CROP_REL = 0.36   # crop size around center relative to min(w,h)
FALLBACK_PARAM2 = 16       # Hough param2 (lower -> more detections)


def _center_of_frame(frame):
    h, w = frame.shape[:2]
    return (w * 0.5, h * 0.5)


def pick_shooter_balls_from_detected(balls, frame_shape, max_dist_rel=SHOOTER_MAX_DIST_REL):
    """
    balls: list of dicts: {"x","y","radius","color"}
    Returns (current_ball, back_ball, debug_info_dict) or (None,None,debug)
      - back_ball: the next ball on frog's back (usually closest to center)
      - current_ball: the ball in mouth (usually 2nd closest to center)
    """
    h, w = frame_shape[:2]
    cx, cy = _center_of_frame(np.zeros((h, w, 3), dtype=np.uint8))
    max_dist = max_dist_rel * min(w, h)

    cand = []
    for b in balls:
        dx = b["x"] - cx
        dy = b["y"] - cy
        d = math.hypot(dx, dy)
        if d <= max_dist:
            cand.append((d, b))

    cand.sort(key=lambda t: t[0])

    debug = {
        "center": (cx, cy),
        "max_dist": max_dist,
        "candidates": cand,  # list of (dist, ball)
    }

    if len(cand) < 2:
        return None, None, debug

    # Closest to center is usually the BACK/NEXT ball.
    back_ball = cand[0][1]
    current_ball = cand[1][1]
    return current_ball, back_ball, debug


def fallback_detect_shooter_balls(frame, detect_fn, classify_fn, balls_hint=None):
    """
    Fallback: run circle detection only in a crop around the center.
    You must pass:
      - detect_fn(crop_bgr) -> circles Nx3 (x,y,r) in crop coords
      - classify_fn(crop_bgr, circles) -> list[{"x","y","radius","color"}] in crop coords
    """
    h, w = frame.shape[:2]
    cx, cy = _center_of_frame(frame)
    ref = min(w, h)

    half = int(max(20, FALLBACK_CROP_REL * ref))
    x0 = max(0, int(cx - half))
    x1 = min(w, int(cx + half))
    y0 = max(0, int(cy - half))
    y1 = min(h, int(cy + half))

    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None, None, {"reason": "empty_crop"}

    circles = detect_fn(crop)
    crop_balls = classify_fn(crop, circles)

    # Shift crop coords -> full-frame coords
    for b in crop_balls:
        b["x"] += x0
        b["y"] += y0

    # Now reuse the “pick closest to center”
    current_ball, back_ball, debug = pick_shooter_balls_from_detected(
        crop_balls, frame.shape, max_dist_rel=SHOOTER_MAX_DIST_REL
    )
    debug["crop_box"] = (x0, y0, x1, y1)
    debug["crop_found"] = len(crop_balls)
    return current_ball, back_ball, debug


def detect_shooter_balls(frame, all_balls, *, detect_fn=None, classify_fn=None):
    """
    Primary: pick from your already detected balls.
    Secondary: fallback center-crop detection.

    Returns:
      current_ball, back_ball, debug
    """
    current_ball, back_ball, debug = pick_shooter_balls_from_detected(all_balls, frame.shape)

    if current_ball is not None and back_ball is not None:
        debug["method"] = "from_global"
        return current_ball, back_ball, debug

    # Optional fallback if you pass detect_fn/classify_fn from your detect module
    if detect_fn is not None and classify_fn is not None:
        current_ball, back_ball, debug2 = fallback_detect_shooter_balls(
            frame, detect_fn, classify_fn, balls_hint=all_balls
        )
        debug2["method"] = "fallback_crop"
        return current_ball, back_ball, debug2

    debug["method"] = "failed_no_fallback"
    return None, None, debug


def draw_shooter_debug(frame, current_ball, back_ball, debug):
    out = frame.copy()
    cx, cy = debug.get("center", _center_of_frame(frame))
    cv.circle(out, (int(cx), int(cy)), 3, (255, 255, 255), -1)
    cv.putText(out, "center", (int(cx) + 6, int(cy) - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    if "crop_box" in debug:
        x0, y0, x1, y1 = debug["crop_box"]
        cv.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1)

    def _draw_ball(ball, name, color_bgr):
        if ball is None:
            return
        x, y, r = int(ball["x"]), int(ball["y"]), int(ball["radius"])
        cv.circle(out, (x, y), r, color_bgr, 2)
        cv.putText(out, f"{name}:{ball['color']}", (x - r, y - r - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv.LINE_AA)

    _draw_ball(back_ball, "back", (0, 255, 255))
    _draw_ball(current_ball, "curr", (0, 255, 0))

    return out