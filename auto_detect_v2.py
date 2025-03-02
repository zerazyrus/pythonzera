import os
import time
import cv2
import numpy as np
import mss
import pyautogui
import keyboard
import json

config_path = "config.json"
save_dir = "sources/images"
os.makedirs(save_dir, exist_ok=True)

padding = 10  # Khoảng dư khi tìm ảnh
coords = []


def load_config():
    """Đọc cấu hình từ file config.json"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Lỗi đọc file config.json, sẽ tạo lại file mới.")
                return {}
    return {}

def save_config(data):
    """Lưu cấu hình vào file config.json"""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def on_mouse(event, x, y, flags, param):
    global coords
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        coords.append((x, y))

def select_region():
    global coords
    coords = []
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    cv2.namedWindow("Chọn vùng tìm kiếm", cv2.WINDOW_NORMAL)
    cv2.imshow("Chọn vùng tìm kiếm", img)
    cv2.setMouseCallback("Chọn vùng tìm kiếm", on_mouse)
    
    while len(coords) < 2:
        if keyboard.is_pressed("esc"):
            cv2.destroyAllWindows()
            print("❌ Đã hủy chọn vùng.")
            return None
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    print(f"✅ Vùng tìm kiếm: ({x1}, {y1}) -> ({x2}, {y2})")
    
    return {
        'left': max(0, x1 - padding),
        'top': max(0, y1 - padding),
        'width': max(1, abs(x2 - x1) + 2 * padding),
        'height': max(1, abs(y2 - y1) + 2 * padding)
    }

def select_image():
    """Chọn ảnh cần tìm kiếm."""
    global coords
    coords = []

    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    cv2.namedWindow("Chọn vùng ảnh", cv2.WINDOW_NORMAL)
    cv2.imshow("Chọn vùng ảnh", img)
    cv2.setMouseCallback("Chọn vùng ảnh", on_mouse)

    while len(coords) < 2:
        if keyboard.is_pressed("esc"):
            cv2.destroyAllWindows()
            print("❌ Đã hủy chọn ảnh.")
            return None
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    cropped_img = img[y1:y2, x1:x2]

    file_index = 1
    while os.path.exists(os.path.join(save_dir, f"target_{file_index}.png")):
        file_index += 1

    image_path = os.path.join(save_dir, f"target_{file_index}.png")
    cv2.imwrite(image_path, cropped_img)
    print(f"✅ Đã lưu ảnh: {image_path}")
    return image_path

def capture_screen(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def find_image(prefix, region, threshold=0.9):
    screen = capture_screen(region)
    cv2.imwrite("debug_screen.png", screen)

    # Nếu ảnh chụp màn hình có 4 kênh (RGBA), chuyển về BGR
    if screen.shape[-1] == 4:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    
    for image_file in os.listdir(save_dir):
        if image_file.startswith(prefix):
            image_path = os.path.join(save_dir, image_file)
            target = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
              # Kiểm tra xem ảnh có hợp lệ không
            if target is None or target.size == 0:
                print(f"⚠️ Không thể đọc ảnh {image_path}, bỏ qua.")
                continue

            # Nếu ảnh target có 4 kênh (RGBA), chuyển về BGR
            if target.shape[-1] == 4:
                target = cv2.cvtColor(target, cv2.COLOR_BGRA2BGR)
            
            result = cv2.matchTemplate(screen, target, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # TM_SQDIFF_NORMED: Giá trị nhỏ hơn mới là kết quả tốt
            if min_val < 0.1:  
                return (region['left'] + min_loc[0], region['top'] + min_loc[1])
    return None

def click_on_image(region):
    pyautogui.tripleClick(region[0] + 5, region[1] + 5)
    print(f"🖱 Click tại ({region[0]}, {region[1]})")
    return
        
def click_on_action(action):
    if action == "click":
        pyautogui.tripleClick()
        print(f"🖱 Click ({action})")
    else :
        keyboard.press(action)
        print(f"🖱 Click ({action})")
        return

def dropFishingRod(region, timeout=0.5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        pos = find_image('balo', region, threshold=0.9)
        if pos:
            time.sleep(2)
            click_on_action("click")
            print("🎣 Đã thả cần câu!")
            return True
    print("⏳ Không tìm thấy balo.")
    return False

def pullFishingRod(region):
    while True:  # Vòng lặp chạy vô hạn
        pos = find_image('signal', region, threshold=0.9)
        if pos:
            click_on_image(pos)
            print("🎣 Đã kéo cần câu!")
            return  # Thoát vòng lặp khi thành công
        time.sleep(0.1)  # Giảm tải CPU, tránh spam tìm kiếm quá nhanh
        
# 🛠 Kiểm tra và load cấu hình
config = load_config()

# Đảm bảo có key "region_search" trong config
if "region_search" not in config:
    config["region_search"] = {}

# Kiểm tra và tải các vùng tìm kiếm
for key in ["region_search_fishing", "region_search_balo"]:
    if key in config["region_search"]:
        print(f"✅ Đã tải {key} từ config: {config['region_search'][key]}")
    else:
        print(f"🔍 Chọn vùng cần tìm kiếm cho {key}.")
        searchRegionDetail = None
        while searchRegionDetail is None:
            searchRegionDetail = select_region()
        
        # Lưu vào config
        config["region_search"][key] = searchRegionDetail
        save_config(config)
        print(f"💾 Đã lưu {key} vào {config_path}.")

# **Bước 2: Hiển thị menu lựa chọn**
while True:
    print("\n🎯 Chọn một tùy chọn:")
    print("1️⃣ Nhấn ENTER để chọn ảnh")
    print("2️⃣ Nhấn SPACE để chạy Auto Click")
    print("❌ Nhấn ESC để thoát")

    key = keyboard.read_event().name  
    if key == "esc":
        print("🚪 Thoát chương trình.")
        break
    if key == "enter":
        select_image()
    if key == "space":
        print("🚀 Bắt đầu Auto Click...")
        break

print("🔍 Đang tìm ảnh... Nhấn ESC để dừng Auto Click.")
paused = False  # Trạng thái tạm dừng

while True:
    if keyboard.is_pressed("esc"):
        print("🛑 Dừng Auto Click.")
        break

    if keyboard.is_pressed("p"):
        paused = not paused  # Đảo trạng thái tạm dừng
        if paused:
            print("⏸ Auto Click đã tạm dừng. Nhấn 'P' để tiếp tục.")
        else:
            print("▶ Tiếp tục Auto Click.")
        time.sleep(1)  # Tránh spam phím

    if keyboard.is_pressed("b"):
        print("🔙 Quay lại chọn ảnh.")
        select_image()  # Quay lại bước chọn ảnh
        time.sleep(1)  # Tránh spam phím

    if not paused:
        success = dropFishingRod(config["region_search"]["region_search_balo"], timeout=0.5)
        if success:
            time.sleep(1)
            pullFishingRod(config["region_search"]["region_search_fishing"])
            time.sleep(2)