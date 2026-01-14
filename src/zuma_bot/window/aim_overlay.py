import math
import cv2

def draw_aim_overlay(
    frame,
    center_xy,
    aim_angle_deg,
    *,
    tick_every_deg=15,
    ring_radius=70,
    ray_length=260,
    draw_ticks=True
):
    """
    Draw a 360° ring of tick arrows + a highlighted aim ray.
    aim_angle_deg: 0..360, where 0 = right, 90 = up, 180 = left, 270 = down.
    """
    out = frame
    cx, cy = int(center_xy[0]), int(center_xy[1])

    # --- helper to convert angle->direction (note: y axis down in images)
    def dir_from_angle(deg):
        rad = math.radians(deg)
        dx = math.cos(rad)
        dy = -math.sin(rad)
        return dx, dy

    # --- optional tick arrows around 360
    if draw_ticks and tick_every_deg > 0:
        for a in range(0, 360, tick_every_deg):
            dx, dy = dir_from_angle(a)

            # short arrow on ring
            x1 = int(cx + dx * ring_radius)
            y1 = int(cy + dy * ring_radius)
            x2 = int(cx + dx * (ring_radius + 18))
            y2 = int(cy + dy * (ring_radius + 18))

            cv2.arrowedLine(out, (x1, y1), (x2, y2), (180, 180, 180), 1, tipLength=0.35)

    # --- highlighted aim ray
    dx, dy = dir_from_angle(aim_angle_deg)
    ex = int(cx + dx * ray_length)
    ey = int(cy + dy * ray_length)

    cv2.arrowedLine(out, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.06)

    # endpoint marker (use later for sampling)
    cv2.circle(out, (ex, ey), 4, (0, 255, 255), -1)

    # center marker + angle text
    cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)
    cv2.putText(out, f"aim={aim_angle_deg:.1f} deg", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    return out, (ex, ey)
