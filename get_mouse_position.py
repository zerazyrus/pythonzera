import cv2
import numpy as np
import mss

coords = []

def on_mouse(event, x, y, flags, param):
    global coords
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        coords.append((x, y))
        cv2.destroyAllWindows()

def select_region():
    global coords
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Chọn vùng (Giữ chuột trái và kéo)", img)
    cv2.setMouseCallback("Chọn vùng (Giữ chuột trái và kéo)", on_mouse)
    cv2.waitKey(0)

    if len(coords) == 2:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        return {'left': x1, 'top': y1, 'width': x2 - x1, 'height': y2 - y1}
    return None

# Chạy chọn vùng
region = select_region()
print(f"Vùng đã chọn: {region}")
