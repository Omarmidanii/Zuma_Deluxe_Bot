import cv2 as cv
import numpy as np

img = cv.imread("assets/templates/image.png")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        h, s, v = hsv[y, x]
        print(f"x={x} y={y} -> H={int(h)} S={int(s)} V={int(v)}")

cv.imshow("img", img)
cv.setMouseCallback("img", on_mouse)
cv.waitKey(0)   
cv.destroyAllWindows()
