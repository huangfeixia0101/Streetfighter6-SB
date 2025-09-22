import cv2
import numpy as np
import tqdm
import csv
import os

VIDEO = r"C:\Users\huang\Videos\SF6\SF6.mp4"
ALPHA = 0.08

def hp_ratio(roi):
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red  = cv2.inRange(hsv, (165, 250, 115), (170, 255, 210))
    mask_blue = cv2.inRange(hsv, (100, 200, 130), (115, 225, 190))
    mask_yellow = cv2.inRange(hsv, (25, 50, 240), (35, 150, 255))
    mask = mask_red | mask_blue | mask_yellow
    col_flag = np.sum(mask > 0, axis=0) >= max(1, roi.shape[0] // 10)
    return np.count_nonzero(col_flag) / len(col_flag), mask

def xyz_off_scan():
    cap = cv2.VideoCapture(VIDEO)
    total = 100                      # 只读前 100 帧
    best_cfg = None
    best_score = -1.0

    for x in range(185 - 5, 185 + 6):          # x=180→190
        for y in range(68 - 5, 68 + 6):        # y=63→73
            for w in range(664 - 5, 664 + 6):  # w=659→669
                for off in range(886 - 5, 886 + 6):  # off=881→891
                    roi_left  = (x, y, w, 24)
                    roi_right = (x + off, y, w, 24)   # 左右偏移可调
                    max_left = max_right = 0.0

                    for idx in tqdm.trange(total, desc=f"x={x},y={y},w={w},off={off}"):
                        ret, frame = cap.read()
                        if not ret: break
                        left_r,  _  = hp_ratio(frame[roi_left[1]:roi_left[1]+roi_left[3], roi_left[0]:roi_left[0]+roi_left[2]])
                        right_r, _  = hp_ratio(frame[roi_right[1]:roi_right[1]+roi_right[3], roi_right[0]:roi_right[0]+roi_right[2]])
                        max_left  = max(max_left, left_r)
                        max_right = max(max_right, right_r)

                    score = max(max_left, max_right)
                    print(f"x={x},y={y},w={w},off={off}: 双侧最大 = {score:.4f}  (left={max_left:.4f}, right={max_right:.4f})")

                    if score > best_score:
                        best_score = score
                        best_cfg = (x, y, w, off, max_left, max_right)

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 复位视频指针

    cap.release()
    if best_cfg:
        x, y, w, off, l, r = best_cfg
        print("\n=== 最优四维度坐标 ===")
        print(f"ROI_LEFT  = ({x}, {y}, {w}, 24)")
        print(f"ROI_RIGHT = ({x + off}, {y}, {w}, 24)")
        print(f"偏移量 = {off}")
        print(f"双侧最大值 = {best_score:.4f}  (left={l:.4f}, right={r:.4f})")
    else:
        print("未找到任何 CSV 文件！")

print("=== 开始四维度网格扫描 (x±5, y±5, w±5, offset±5) ===")
xyz_off_scan()
print("全部 14641 组扫描完成！")