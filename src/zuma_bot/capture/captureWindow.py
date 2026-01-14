from __future__ import annotations

import math
import time
import ctypes
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mss

# from window.feature_window_detector import FeatureWindowDetector
from window.multi_level_window_detector import MultiLevelWindowDetector
import window.detect_balls as wd
import window.detect_shooter as shooter
import window.aim_overlay as aim


ROI_WINDOW = "Zuma ROI + Balls"
ANCHOR_WINDOW = "Zuma Anchor Debug"
DRAW_RAYS = False
HIGHLIGHT_CHOSEN = True
SHOW_ANCHOR_DEBUG = True
TARGET_FPS = 30
ANCHOR_EVERY_N = 12
BALL_EVERY_N = 2
BBOX_PAD = 12
USE_RIGHT_HALF = True
AUTO_PLAY = True
SHOT_COOLDOWN_SEC = 0.40
RAY_STEP_DEG = 8
TARGET_STABLE_FRAMES = 2
EXCLUDE_CENTER_REL = getattr(shooter, "SHOOTER_MAX_DIST_REL", 0.24)
NEIGHBOR_LINK_FACTOR = 1.28
SAME_COLOR_LINK_FACTOR = 1.18


def _init_dpi_awareness() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_init_dpi_awareness()
_user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

CLICK_HOLD_SEC = 0.03


def click_at_screen(x: float, y: float) -> None:
    x = int(round(x))
    y = int(round(y))
    _user32.SetCursorPos(x, y)
    _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(CLICK_HOLD_SEC)
    _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def highlight_target_ball(img: np.ndarray, target: Optional[dict], label: str = "TARGET") -> np.ndarray:
    if target is None:
        return img

    x = int(round(target["x"]))
    y = int(round(target["y"]))
    r = int(round(target.get("radius", 10)))

    cv2.circle(img, (x, y), r + 4, (0, 0, 255), 3)
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    cv2.putText(
        img,
        label,
        (x + r + 8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _ray_end_at_border(cx: float, cy: float, dx: float, dy: float, w: int, h: int) -> Tuple[int, int]:
    eps = 1e-9
    ts: List[float] = []

    if abs(dx) > eps:
        t1 = (0 - cx) / dx
        t2 = ((w - 1) - cx) / dx
        if t1 > 0:
            ts.append(t1)
        if t2 > 0:
            ts.append(t2)

    if abs(dy) > eps:
        t3 = (0 - cy) / dy
        t4 = ((h - 1) - cy) / dy
        if t3 > 0:
            ts.append(t3)
        if t4 > 0:
            ts.append(t4)

    if not ts:
        return int(cx), int(cy)

    t = min(ts)
    ex = cx + dx * t
    ey = cy + dy * t
    return int(round(ex)), int(round(ey))


def draw_rays_from_origin(img: np.ndarray, origin_ball: dict, step_deg: int = 5, thickness: int = 1) -> np.ndarray:
    out = img
    h, w = out.shape[:2]
    cx = float(origin_ball["x"])
    cy = float(origin_ball["y"])

    cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    cv2.putText(out, "origin", (int(cx) + 8, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for a in range(0, 360, int(step_deg)):
        rad = math.radians(a)
        dx = math.cos(rad)
        dy = -math.sin(rad)
        ex, ey = _ray_end_at_border(cx, cy, dx, dy, w, h)
        cv2.line(out, (int(cx), int(cy)), (ex, ey), (255, 255, 255), thickness, cv2.LINE_AA)

    return out


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]

    def size(self, a: int) -> int:
        return self.sz[self.find(a)]


def _center_of_frame(frame_shape: Tuple[int, int, int]) -> Tuple[float, float]:
    h, w = frame_shape[:2]
    return (w * 0.5, h * 0.5)


def split_track_vs_shooter_balls(
    balls: List[dict],
    frame_shape: Tuple[int, int, int],
    *,
    exclude_center_rel: float = EXCLUDE_CENTER_REL,
) -> Tuple[List[dict], List[dict]]:
    """Split detected circles into track balls vs shooter-area balls."""
    if not balls:
        return [], []

    h, w = frame_shape[:2]
    cx, cy = _center_of_frame(frame_shape)
    center_cut = float(exclude_center_rel) * float(min(w, h))

    track: List[dict] = []
    shooter_area: List[dict] = []
    for b in balls:
        d = math.hypot(float(b["x"]) - cx, float(b["y"]) - cy)
        if d < center_cut:
            shooter_area.append(b)
        else:
            track.append(b)
    return track, shooter_area


def compute_chain_neighbor_edges(
    track_balls: List[dict],
    *,
    neighbor_factor: float = NEIGHBOR_LINK_FACTOR,
    max_neighbors: int = 2,
) -> List[Tuple[int, int]]:
    """
    Approximate the 1D chain by giving each ball up to `max_neighbors` closest neighbors
    that are within (r_i + r_j) * neighbor_factor.

    This avoids the common failure mode of 2D clustering where balls on different parts
    of a curved track get merged just because they're close on screen.
    """
    n = len(track_balls)
    if n <= 1:
        return []

    edges: List[Tuple[int, int]] = []
    for i in range(n):
        xi, yi, ri = float(track_balls[i]["x"]), float(track_balls[i]["y"]), float(track_balls[i]["radius"])
        cands: List[Tuple[float, int]] = []
        for j in range(n):
            if j == i:
                continue
            xj, yj, rj = float(track_balls[j]["x"]), float(track_balls[j]["y"]), float(track_balls[j]["radius"])
            dx = xi - xj
            dy = yi - yj
            dist2 = dx * dx + dy * dy
            thresh = (ri + rj) * float(neighbor_factor)
            if dist2 <= thresh * thresh:
                cands.append((dist2, j))

        cands.sort(key=lambda t: t[0])
        for _, j in cands[:max_neighbors]:
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b))

    # Deduplicate
    edges = sorted(set(edges))
    return edges


def compute_same_color_cluster_sizes(
    track_balls: List[dict],
    *,
    neighbor_edges: Optional[List[Tuple[int, int]]] = None,
    same_color_factor: float = SAME_COLOR_LINK_FACTOR,
) -> Dict[int, int]:
    """
    Return dict id(ball) -> size of its SAME-COLOR connected component.

    We only connect SAME-COLOR balls if:
      1) They are chain-neighbors (from `neighbor_edges`), AND
      2) Their distance is within a slightly tighter threshold.

    This tends to correlate better with real "adjacent in chain" runs.
    """
    n = len(track_balls)
    if n == 0:
        return {}

    if neighbor_edges is None:
        neighbor_edges = compute_chain_neighbor_edges(track_balls)

    dsu = _DSU(n)

    for (i, j) in neighbor_edges:
        bi = track_balls[i]
        bj = track_balls[j]
        if bi.get("color") != bj.get("color"):
            continue

        xi, yi, ri = float(bi["x"]), float(bi["y"]), float(bi["radius"])
        xj, yj, rj = float(bj["x"]), float(bj["y"]), float(bj["radius"])
        dx = xi - xj
        dy = yi - yj
        dist2 = dx * dx + dy * dy
        thresh = (ri + rj) * float(same_color_factor)
        if dist2 <= thresh * thresh:
            dsu.union(i, j)

    out: Dict[int, int] = {}
    for i in range(n):
        out[id(track_balls[i])] = int(dsu.size(i))
    return out


def first_hit_ball_per_ray(
    origin_ball: dict,
    balls: List[dict],
    *,
    step_deg: int = 5,
) -> List[dict]:
    """Cast rays from origin_ball; return first hit per ray."""
    if origin_ball is None or not balls:
        return []

    ox = float(origin_ball["x"])
    oy = float(origin_ball["y"])
    orad = float(origin_ball.get("radius", 0))

    hits: List[dict] = []
    step_deg = max(1, int(step_deg))

    for ang in range(0, 360, step_deg):
        rad = math.radians(ang)
        ux = math.cos(rad)
        uy = -math.sin(rad)

        best_t: Optional[float] = None
        best_ball: Optional[dict] = None

        for b in balls:
            cx = float(b["x"])
            cy = float(b["y"])
            r = float(b["radius"])

            vx = cx - ox
            vy = cy - oy

            tca = vx * ux + vy * uy
            if tca < 0:
                continue

            d2 = (vx * vx + vy * vy) - (tca * tca)
            rr = r * r
            if d2 > rr:
                continue

            thc = math.sqrt(max(0.0, rr - d2))
            t0 = tca - thc
            t1 = tca + thc

            t_hit: Optional[float] = None
            if t0 > 1e-6:
                t_hit = t0
            elif t1 > 1e-6:
                t_hit = t1

            if t_hit is None:
                continue

            # Ignore tiny self-intersections / noise
            if t_hit < max(1.0, 0.25 * orad):
                continue

            if best_t is None or t_hit < best_t:
                best_t = t_hit
                best_ball = b

        if best_ball is not None and best_t is not None:
            hits.append({"angle": ang, "ball": best_ball, "t_hit": float(best_t)})

    return hits


def unique_hits_by_ball(hits: List[dict]) -> List[dict]:
    """Collapse many angle-hits into one candidate per ball."""
    best_by_id: Dict[int, dict] = {}
    for h in hits:
        b = h["ball"]
        bid = id(b)
        prev = best_by_id.get(bid)
        if prev is None or h["t_hit"] < prev["t_hit"]:
            best_by_id[bid] = h
    return list(best_by_id.values())


def find_empty_angle(origin_ball: dict, balls: List[dict], *, step_deg: int = 5) -> int:
    """Return an angle (deg) that doesn't intersect any of the provided balls, if possible."""
    hits = first_hit_ball_per_ray(origin_ball, balls, step_deg=step_deg)
    hit_angles = {h["angle"] for h in hits}

    step_deg = max(1, int(step_deg))
    for ang in range(0, 360, step_deg):
        if ang not in hit_angles:
            return ang

    if hits:
        # No empty ray — pick the least-bad one (farthest first hit)
        return int(max(hits, key=lambda z: z["t_hit"])["angle"])
    return 0


# ===========================
# Planning
# ===========================


@dataclass
class ShotPlan:
    action: str  # "shoot" | "dump"
    aim_xy: Tuple[float, float]
    why: str
    target: Optional[dict] = None
    target_key: Optional[Tuple] = None


def _ball_key(b: dict) -> Tuple:
    # A stable-ish key across frames.
    return (int(b["x"]), int(b["y"]), int(b.get("radius", 0)), str(b.get("color")))


def plan_best_shot(
    *,
    curr_color: Optional[str],
    next_color: Optional[str],
    origin_ball: Optional[dict],
    track_balls: List[dict],
    frame_shape: Tuple[int, int, int],
    step_deg: int = RAY_STEP_DEG,
) -> ShotPlan:
    """Choose the best action for the current frame (NO SWAP)."""
    h, w = frame_shape[:2]

    if origin_ball is None:
        return ShotPlan(action="dump", aim_xy=(w * 0.5, h * 0.5), why="no_origin")

    # Visibility candidates
    hits = unique_hits_by_ball(first_hit_ball_per_ray(origin_ball, track_balls, step_deg=step_deg))

    # Cluster sizes (same-color runs)
    neighbor_edges = compute_chain_neighbor_edges(track_balls)
    cluster_sizes = compute_same_color_cluster_sizes(track_balls, neighbor_edges=neighbor_edges)

    def aim_xy_for_hit(hit: dict) -> Tuple[float, float]:
        """Click along the SAME angle that produced this first-hit."""
        ang = float(hit["angle"])
        rad = math.radians(ang)
        dx = math.cos(rad)
        dy = -math.sin(rad)  # y axis points downward in image coords

        ox = float(origin_ball["x"])
        oy = float(origin_ball["y"])

        ex, ey = _ray_end_at_border(ox, oy, dx, dy, w, h)

        # optional safety margin so we don't click exactly on the border pixels
        M = 3
        ex = max(M, min(w - 1 - M, ex))
        ey = max(M, min(h - 1 - M, ey))

        return float(ex), float(ey)

    
    def score_pop(hit: dict) -> float:
        b = hit["ball"]
        sz = float(cluster_sizes.get(id(b), 1))
        t = float(hit["t_hit"])
        # Prefer larger clusters, but slightly prefer closer shots (more reliable)
        return (2000.0 if sz >= 2 else 0.0) + 50.0 * sz - 0.10 * t

    def best_hit_for_color(color: str, *, require_cluster_ge: int = 2) -> Optional[dict]:
        cands = []
        for hit in hits:
            b = hit["ball"]
            if b.get("color") != color:
                continue
            sz = cluster_sizes.get(id(b), 1)
            if sz < require_cluster_ge:
                continue
            cands.append(hit)
        if not cands:
            return None
        return max(cands, key=score_pop)

    # 1) POP now with current color
    if curr_color:
        best = best_hit_for_color(curr_color, require_cluster_ge=2)
        if best is not None:
            b = best["ball"]
            sz = cluster_sizes.get(id(b), 1)
            return ShotPlan(
                action="shoot",
                aim_xy=(float(b["x"]), float(b["y"])),
                why=f"POP {curr_color} (cluster={sz})",
                target=b,
                target_key=_ball_key(b),
            )

    # # 2) SETUP: if current==next, grow a single into a pair for the next shot
    # # (You shoot curr into a lone same-color ball => pair; next becomes curr => you pop.)
    # if curr_color and next_color and curr_color == next_color:
    #     setup_cands = []
    #     for hit in hits:
    #         b = hit["ball"]
    #         if b.get("color") != curr_color:
    #             continue
    #         if cluster_sizes.get(id(b), 1) != 1:
    #             continue
    #         setup_cands.append(hit)
    #     if setup_cands:
    #         best = min(setup_cands, key=lambda hh: hh["t_hit"])
    #         b = best["ball"]
    #         return ShotPlan(
    #             action="shoot",
    #             aim_xy=(float(b["x"]), float(b["y"])),
    #             why=f"SETUP {curr_color}→pair (next same)",
    #             target=b,
    #             target_key=_ball_key(b),
    #         )

    # 3) NEXT POP SOON: if next ball has an immediate pop, dump current as safely as possible
    # if next_color and next_color != curr_color:
    #     best_next = best_hit_for_color(next_color, require_cluster_ge=2)
    #     if best_next is not None:
    #         ang = find_empty_angle(origin_ball, track_balls, step_deg=step_deg)
    #         rad = math.radians(ang)
    #         dx = math.cos(rad)
    #         dy = -math.sin(rad)
    #         ex, ey = _ray_end_at_border(float(origin_ball["x"]), float(origin_ball["y"]), dx, dy, w, h)
    #         return ShotPlan(action="dump", aim_xy=(float(ex), float(ey)), why=f"DUMP to reach NEXT POP {next_color}")

    # 4) SETUP anyway: if we can't pop, at least grow a lone ball into a pair
    # (This is better than random shots; eventually you get this color again.)
    if curr_color:
        setup_cands = []
        for hit in hits:
            b = hit["ball"]
            if b.get("color") != curr_color:
                continue
            if cluster_sizes.get(id(b), 1) != 1:
                continue
            setup_cands.append(hit)
        if setup_cands:
            best = min(setup_cands, key=lambda hh: hh["t_hit"])
            b = best["ball"]
            return ShotPlan(
                action="shoot",
                aim_xy=(float(b["x"]), float(b["y"])),
                why=f"SETUP {curr_color}→pair",
                target=b,
                target_key=_ball_key(b),
            )
    print("empty")
    # 5) DUMP: shoot into empty space (or the farthest first-hit ray)
    ang = find_empty_angle(origin_ball, track_balls, step_deg=step_deg)
    rad = math.radians(ang)
    dx = math.cos(rad)
    dy = -math.sin(rad)
    ex, ey = _ray_end_at_border(float(origin_ball["x"]), float(origin_ball["y"]), dx, dy, w, h)
    return ShotPlan(action="dump", aim_xy=(float(ex), float(ey)), why=f"DUMP (ang={ang})")


# ===========================
# ROI helpers
# ===========================


def clamp_region(mon: dict, left: int, top: int, width: int, height: int) -> dict:
    ml, mt = int(mon["left"]), int(mon["top"])
    mr, mb = ml + int(mon["width"]), mt + int(mon["height"])

    left = int(max(ml, left))
    top = int(max(mt, top))
    right = int(min(mr, left + width))
    bottom = int(min(mb, top + height))

    width = int(max(1, right - left))
    height = int(max(1, bottom - top))
    return {"left": left, "top": top, "width": width, "height": height}


def right_half(mon: dict) -> dict:
    return {
        "left": int(mon["left"] + mon["width"] // 2),
        "top": int(mon["top"]),
        "width": int(mon["width"] // 2),
        "height": int(mon["height"]),
    }


def region_from_bbox(mon: dict, base_region: dict, bbox, pad: int = 0) -> dict:
    left = int(base_region["left"] + bbox.x - pad)
    top = int(base_region["top"] + bbox.y - pad)
    width = int(bbox.w + 2 * pad)
    height = int(bbox.h + 2 * pad)
    return clamp_region(mon, left, top, width, height)


# ===========================
# Main loop
# ===========================


def RunCapture() -> None:
    paused = False
    frame_i = 0
    aim_angle_deg = 0.0
    AIM_STEP = 2.0

    last_bbox = None
    last_anchor_frame = None
    last_balls: List[dict] = []
    last_annotated = None
    last_roi_frame = None

    # State for planning / debouncing
    virtual_curr_color: Optional[str] = None
    last_shot_time = 0.0

    last_plan_key = None
    stable_count = 0

    cv2.namedWindow(ROI_WINDOW, cv2.WINDOW_NORMAL)
    if SHOW_ANCHOR_DEBUG:
        cv2.namedWindow(ANCHOR_WINDOW, cv2.WINDOW_NORMAL)

    # det = FeatureWindowDetector("../../assets/templates/level2.png")
    det = MultiLevelWindowDetector([
    ("level1", "../../assets/templates/level1.png"),
    ("level2", "../../assets/templates/level2.png"),
    ("level3", "../../assets/templates/level3.png"),
    ("level4", "../../assets/templates/level4.png"),
])

    with mss.mss() as sct:
        mon = sct.monitors[1]

        anchor_region = right_half(mon) if USE_RIGHT_HALF else {
            "left": int(mon["left"]),
            "top": int(mon["top"]),
            "width": int(mon["width"]),
            "height": int(mon["height"]),
        }

        while True:
            t0 = time.perf_counter()

            if not paused:
                # 1) Anchor detection every N frames
                need_anchor = (last_bbox is None) or (frame_i % ANCHOR_EVERY_N == 0)
                if need_anchor:
                    raw = np.asarray(sct.grab(anchor_region))
                    anchor_frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

                    found = det.locate_anchor(anchor_frame)
                    if found:
                        bbox, inlier_ratio, quad = found
                        last_bbox = bbox
                        x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
                        cv2.rectangle(anchor_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if quad is not None:
                            cv2.polylines(anchor_frame, [quad.astype(int)], True, (0, 255, 0), 2)
                        cv2.putText(
                            anchor_frame,
                            f"inlier_ratio={inlier_ratio:.2f}",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                    last_anchor_frame = anchor_frame

                # 2) ROI capture
                if last_bbox is not None:
                    roi_region = region_from_bbox(mon, anchor_region, last_bbox, pad=BBOX_PAD)
                else:
                    roi_region = anchor_region

                raw_roi = np.asarray(sct.grab(roi_region))
                roi_frame = cv2.cvtColor(raw_roi, cv2.COLOR_BGRA2BGR)
                last_roi_frame = roi_frame

                # 3) Ball detection every N frames
                if frame_i % BALL_EVERY_N == 0:
                    last_balls, last_annotated = wd.start_ball_detection(
                        roi_frame,
                        show=False,
                        debug=False,
                        return_annotated=True,
                    )

                # 4) Shooter balls (current + back)
                det_curr, det_back, sh_debug = shooter.detect_shooter_balls(
                    roi_frame,
                    last_balls,
                    detect_fn=wd.detect_zuma_balls,
                    classify_fn=wd.classify_circles,
                )

                # Origin (best): mouth ball center; fallback: back ball; fallback: ROI center
                origin_ball = det_curr or det_back
                if origin_ball is None:
                    hh, ww = roi_frame.shape[:2]
                    origin_ball = {"x": ww * 0.5, "y": hh * 0.5, "radius": 10, "color": None}

                # Current color: prefer virtual state (stable across frames), but seed from detection
                if virtual_curr_color is None and det_curr is not None:
                    virtual_curr_color = det_curr.get("color")

                now = time.perf_counter()

                # If we haven't shot recently, trust the detector to re-sync.
                if det_curr is not None and (now - last_shot_time) > 0.60:
                    virtual_curr_color = det_curr.get("color")

                curr_color = virtual_curr_color
                next_color = det_back.get("color") if det_back is not None else None

                # 5) Split targets: track vs shooter area
                track_balls, _shooter_area_balls = split_track_vs_shooter_balls(
                    last_balls,
                    roi_frame.shape,
                    exclude_center_rel=EXCLUDE_CENTER_REL,
                )

                # 6) Plan
                plan = plan_best_shot(
                    curr_color=curr_color,
                    next_color=next_color,
                    origin_ball=origin_ball,
                    track_balls=track_balls,
                    frame_shape=roi_frame.shape,
                    step_deg=RAY_STEP_DEG,
                )

                # Quantize aim so detection jitter doesn't break stability.
                q = 4
                plan_key = (
                    plan.action,
                    plan.target_key,
                    (int(round(plan.aim_xy[0] / q)), int(round(plan.aim_xy[1] / q))),
                )
                if plan_key == last_plan_key:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_plan_key = plan_key

                # 7) Display
                display = last_annotated if last_annotated is not None else roi_frame

                if DRAW_RAYS and origin_ball is not None:
                    display = draw_rays_from_origin(display, origin_ball, step_deg=RAY_STEP_DEG, thickness=1)

                if HIGHLIGHT_CHOSEN and plan.target is not None:
                    display = highlight_target_ball(display, plan.target, label=plan.why)
                else:
                    cv2.putText(
                        display,
                        plan.why,
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                status = f"anchor={'yes' if last_bbox is not None else 'no'}  balls={len(last_balls)}  track={len(track_balls)}"
                if curr_color is not None:
                    status += f"  curr={curr_color}"
                if next_color is not None:
                    status += f"  next={next_color}"
                status += f"  stable={stable_count}/{TARGET_STABLE_FRAMES}"
                cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Aim overlay uses shooter debug center if available
                if sh_debug is not None and "center" in sh_debug:
                    aim_center = sh_debug["center"]
                else:
                    hh, ww = display.shape[:2]
                    aim_center = (ww * 0.5, hh * 0.5)

                display, _ = aim.draw_aim_overlay(
                    display,
                    aim_center,
                    aim_angle_deg,
                    tick_every_deg=15,
                    ring_radius=70,
                    ray_length=260,
                    draw_ticks=True,
                )

                cv2.imshow(ROI_WINDOW, display)
                if SHOW_ANCHOR_DEBUG and last_anchor_frame is not None and need_anchor:
                    cv2.imshow(ANCHOR_WINDOW, last_anchor_frame)

                # 8) Auto-play
                can_shoot = (now - last_shot_time) >= SHOT_COOLDOWN_SEC
                stable_enough = stable_count >= TARGET_STABLE_FRAMES

                if AUTO_PLAY and can_shoot and stable_enough:
                    aim_x, aim_y = plan.aim_xy
                    click_at_screen(roi_region["left"] + aim_x, roi_region["top"] + aim_y)
                    last_shot_time = now

                    # After a shot, the next ball becomes current.
                    if next_color is not None:
                        virtual_curr_color = next_color

                    # Reset stability so we don't double-fire same plan
                    stable_count = 0

                frame_i += 1

            else:
                # paused
                if last_annotated is not None:
                    cv2.imshow(ROI_WINDOW, last_annotated)
                elif last_roi_frame is not None:
                    cv2.imshow(ROI_WINDOW, last_roi_frame)
                if SHOW_ANCHOR_DEBUG and last_anchor_frame is not None:
                    cv2.imshow(ANCHOR_WINDOW, last_anchor_frame)

            elapsed = time.perf_counter() - t0
            delay_ms = max(1, int((1.0 / TARGET_FPS - elapsed) * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF

            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("s"):
                out = last_annotated if last_annotated is not None else last_roi_frame
                if out is not None:
                    cv2.imwrite("assets/debug_screenshot.jpg", out)
                    print("Saved assets/debug_screenshot.jpg")
            if key == ord("a"):
                aim_angle_deg = (aim_angle_deg - AIM_STEP) % 360.0
            if key == ord("d"):
                aim_angle_deg = (aim_angle_deg + AIM_STEP) % 360.0
            if key == ord("z"):
                AIM_STEP = min(20.0, AIM_STEP + 1.0)
            if key == ord("x"):
                AIM_STEP = max(0.5, AIM_STEP - 1.0)

    cv2.destroyAllWindows()
