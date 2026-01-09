import time
import cv2
import numpy as np
import mss
from window.feature_window_detector import FeatureWindowDetector
import window.detect_balls as wd

SHOW_EVERY_N = 4          # display 1 out of N frames (bigger => less spam)
TARGET_FPS = 30           # throttle capture/display
WINDOW_NAME = "Zuma Debug"

def RunCapture():
    paused = False
    last_bbox = None
    frame_i = 0
    last_show = time.perf_counter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    det = FeatureWindowDetector("../../assets/templates/anchor.png")
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        da_balls = []
        while True:
            found = 0
            x=0
            y=0
            w=0
            h=0
            if not paused: 
                img = np.array(sct.grab(monitor))             
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                found = det.locate_anchor(frame)
                if found:
                    bbox, inlier_ratio, quad = found
                    last_bbox = bbox
                    x, y, w, h = last_bbox.x , last_bbox.y, last_bbox.w, last_bbox.h
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.polylines(frame, [quad.astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"inlier_ratio={inlier_ratio:.2f}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if frame_i % SHOW_EVERY_N == 0:
                    H,W,_ = frame.shape
                
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(W, x + w)
                    y2 = min(H, y + h)
                    
                    if x2 > x1 and y2 > y1:
                        cropped = frame[y1:y2, x1:x2].copy()
                        da_balls = wd.start_ball_detection(cropped)
                        
                    now = time.perf_counter()
                    dt = now - last_show
                    wait = max(0.0, (1.0 / TARGET_FPS) - dt)
                    if wait > 0:
                        time.sleep(wait)
                    last_show = time.perf_counter()
                frame_i += 1
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('s'):
                cv2.imwrite("assets/debug_screenshot.jpg", frame)
                print("Saved debug_screenshot.jpg")


    cv2.destroyAllWindows()
