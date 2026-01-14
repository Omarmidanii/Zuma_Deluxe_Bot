from __future__ import annotations

"""
Multi-feature window detector (Option B: per-level templates).

You provide *multiple* reference images (one per level / scene). Each reference
image should show the *full game window* you want to capture (client area).

At runtime, we:
  1) Detect ORB features in the frame and in each template.
  2) Match with BFMatcher using Hamming distance (ORB descriptors are binary).
  3) Filter with Lowe's ratio test (knnMatch k=2).
  4) Estimate homography with RANSAC (cv2.findHomography).
  5) Warp the template's 4 corners into the frame (cv2.perspectiveTransform).
  6) Return an axis-aligned bounding box (BBox) of those warped corners.

Pick the best template by:
  - highest number of RANSAC inliers
  - tie-break by inlier ratio

This is the same technique shown in OpenCV's "Feature Matching + Homography to
find Objects" tutorial.

Usage:

    det = MultiLevelWindowDetector([
        ("level1", "assets/templates/level1_full.png"),
        ("level2", "assets/templates/level2_full.png"),
    ])

    found = det.locate_window(frame_bgr)
    if found:
        bbox, inlier_ratio, quad, which = found
        # bbox is the FULL window bbox in the current frame
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bbox in image coordinates (pixels)."""
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class _Template:
    name: str
    path: str
    gray: np.ndarray
    kp: List[cv2.KeyPoint]
    des: np.ndarray
    corners: np.ndarray  # shape (4,1,2) float32


class MultiLevelWindowDetector:
    def __init__(
        self,
        templates: List[Tuple[str, str]],
        *,
        nfeatures: int = 2000,
        ratio_test: float = 0.75,
        min_good_matches: int = 25,
        ransac_reproj_thresh: float = 5.0,
    ):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_test = float(ratio_test)
        self.min_good_matches = int(min_good_matches)
        self.ransac_reproj_thresh = float(ransac_reproj_thresh)

        self._last_name: Optional[str] = None
        self.templates: List[_Template] = self._load_templates(templates)

    def get_last_template_name(self) -> Optional[str]:
        return self._last_name

    # Backwards-compatible alias
    def get_last_anchor_name(self) -> Optional[str]:
        return self._last_name

    def locate_window(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[BBox, float, np.ndarray, str]]:
        """
        Returns:
          (bbox, inlier_ratio, quad, which_template_name)

        - bbox: axis-aligned bounding box of the warped template corners
        - inlier_ratio: inliers / good_matches
        - quad: warped template corners in frame coords, shape (4,1,2)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kpF, desF = self.orb.detectAndCompute(frame_gray, None)
        if desF is None or kpF is None or len(kpF) < 8:
            return None

        best = None  # (inliers, ratio, bbox, quad, name)
        best_inliers = -1
        best_ratio = -1.0

        for T in self.templates:
            # knnMatch(query=template, train=frame)
            matches = self.matcher.knnMatch(T.des, desF, k=2)
            good = []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < self.ratio_test * n.distance:
                    good.append(m)

            if len(good) < self.min_good_matches:
                continue

            src_pts = np.float32([T.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpF[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thresh
            )
            if H is None or mask is None:
                continue

            inliers = int(mask.ravel().sum())
            ratio = float(inliers) / float(len(good))

            # Warp template corners into the frame
            quad = cv2.perspectiveTransform(T.corners, H)

            # Axis-aligned bbox from quad
            xs = quad[:, 0, 0]
            ys = quad[:, 0, 1]
            x0 = int(np.floor(xs.min()))
            y0 = int(np.floor(ys.min()))
            x1 = int(np.ceil(xs.max()))
            y1 = int(np.ceil(ys.max()))
            bbox = BBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))

            # Choose the best by inliers, then inlier ratio
            if (inliers > best_inliers) or (inliers == best_inliers and ratio > best_ratio):
                best_inliers = inliers
                best_ratio = ratio
                best = (inliers, ratio, bbox, quad, T.name)

        if best is None:
            self._last_name = None
            return None

        _, ratio, bbox, quad, name = best
        self._last_name = name
        return bbox, ratio, quad, name

    # Backwards-compatible alias: "anchor" == "window template" here
    def locate_anchor(self, frame_bgr: np.ndarray):
        out = self.locate_window(frame_bgr)
        if out is None:
            return None
        bbox, ratio, quad, _name = out
        return bbox, ratio, quad

    def _load_templates(self, templates: List[Tuple[str, str]]) -> List[_Template]:
        out: List[_Template] = []
        for name, path in templates:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read template '{name}' from path: {path}")

            kp, des = self.orb.detectAndCompute(img, None)
            if des is None or kp is None or len(kp) < 8:
                raise ValueError(
                    f"Template '{name}' has too few ORB features. "
                    f"Use a larger screenshot / crop with more corners & texture, "
                    f"or increase nfeatures."
                )

            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

            out.append(_Template(name=name, path=path, gray=img, kp=kp, des=des, corners=corners))

        if not out:
            raise ValueError("No templates provided.")
        return out
