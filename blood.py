import cv2
import numpy as np
import tqdm
import csv
import os

# ========= 用户参数 =========
VIDEO       = r"C:\Users\huang\Videos\SF6\SF6.mp4"
OUT_VIDEO   = True          # 是否保存带叠加的视频
OUT_CSV     = True          # 是否导出 CSV
ALPHA       = 0.08          # 指数平滑
MIN_COL_PIX = 2             # 列统计阈值
# ============================

# 固定 1920×1080 血条坐标（高 35px，宽 705px）
ROI_LEFT  = (182, 68, 665, 24)   # x, y, w, h
ROI_RIGHT = (1063, 68, 665, 24)

# === 前 100 帧内 双边≥97 % 的最佳组 ===
# ROI_LEFT  = (182, 68, 670, 24)
# ROI_RIGHT = (1061, 68, 670, 24)
# 偏移量 = 879  |  score=0.9597  (L=0.9791, R=0.9806)

# === 最优四维度坐标 ===
# ROI_LEFT  = (184, 68, 665, 24)
# ROI_RIGHT = (1065, 68, 665, 24)
# 偏移量 = 881
# 双侧最大值 = 0.9594  (left=0.9789, right=0.9805)
# 全部 14641 组扫描完成！


# 98%
# ROI_LEFT  = (189, 68, 660, 24)   # x, y, w, h
# ROI_RIGHT = (1075, 68, 660, 24)

# === 最优四维度坐标 ===
# ROI_LEFT  = (182, 68, 669, 24)
# ROI_RIGHT = (1063, 68, 669, 24)
# 偏移量 = 881
# 双侧最大值 = 0.9596  (left=0.9791, right=0.9806)

def hp_ratio(roi):

    # 取色
    # cv2.imshow("ROI_pick", roi)
    # cv2.setMouseCallback("ROI_pick",
    #     lambda e, x, y, f, p: print("BGR:", roi[y, x], "HSV:",
    #                                  cv2.cvtColor(roi[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0,0]) if e == cv2.EVENT_LBUTTONDOWN else None)
    # cv2.waitKey(0)          # 按任意键继续
    # cv2.destroyWindow("ROI_pick")

    # 1. 轻微高斯去噪
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 红/粉（实测低 H + 低 S）
    mask_red = cv2.inRange(hsv, (160, 250, 115), (175, 255, 210))
    # 蓝（实测高 H + 中高 S）
    mask_blue = cv2.inRange(hsv, (100, 200, 130), (115, 225, 190))
    # 黄/橙（残血）
    mask_yellow = cv2.inRange(hsv, (25, 50, 240), (35, 150, 255))

    mask = mask_red | mask_blue | mask_yellow

    # 3. 动态阈值：每列非零像素 ≥ 行高的 1/10（抗锯齿）
    h, w = mask.shape
    col_sum = np.sum(mask > 0, axis=0)
    col_flag = col_sum >= max(1, h // 10)   # 动态阈值
    cnt = np.count_nonzero(col_flag)
    return (cnt / len(col_flag)) if col_flag.size else 0.0, mask

cap = cv2.VideoCapture(VIDEO)

# # 单帧调试
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 确保首帧
# ret, first = cap.read()
# cv2.imshow("first_frame", first)
# cv2.waitKey(0)
# cv2.destroyWindow("first_frame")
# cap.release()
#
# r = cv2.selectROI('full_hp', first, False)
# print("满血最左侧坐标:", tuple(map(int, r)))

fps   = int(cap.get(cv2.CAP_PROP_FPS))
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO[:-4] + "_overlay.mp4", fourcc, fps, (w, h)) if OUT_VIDEO else None
csv_file = open(VIDEO[:-4] + ".csv", "w", newline='', encoding='utf-8') if OUT_CSV else None
if csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "left", "right"])

sleft = sright = 0

# 调试窗口
cv2.namedWindow("ROI_left", cv2.WINDOW_NORMAL)
cv2.namedWindow("mask_left", cv2.WINDOW_NORMAL)
cv2.namedWindow("SF6", cv2.WINDOW_NORMAL)

for idx in tqdm.trange(total):
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 裁固定 ROI
    left_roi  = frame[ROI_LEFT[1]:ROI_LEFT[1]+ROI_LEFT[3], ROI_LEFT[0]:ROI_LEFT[0]+ROI_LEFT[2]]
    right_roi = frame[ROI_RIGHT[1]:ROI_RIGHT[1]+ROI_RIGHT[3], ROI_RIGHT[0]:ROI_RIGHT[0]+ROI_RIGHT[2]]

    # 2. 计算比例
    left_r,  left_mask  = hp_ratio(left_roi)
    right_r, right_mask = hp_ratio(right_roi)

    # 3. 指数平滑
    sleft  = left_r  if idx == 0 else ALPHA * left_r  + (1 - ALPHA) * sleft
    sright = right_r if idx == 0 else ALPHA * right_r + (1 - ALPHA) * sright

    # 4. 画字
    cv2.putText(frame, f"L:{sleft*100:.0f}%", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
    cv2.putText(frame, f"R:{sright*100:.0f}%", (w - 300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)

    # 5. 调试窗口
    cv2.imshow("ROI_left", left_roi)
    cv2.imshow("mask_left", left_mask)
    cv2.imshow("SF6", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 6. 写文件
    if out:
        out.write(frame)
    if csv_file:
        writer.writerow([idx, round(sleft, 4), round(sright, 4)])

cap.release()
if out:
    out.release()
if csv_file:
    csv_file.close()
cv2.destroyAllWindows()

print("Done! 输出文件：")
if OUT_VIDEO:
    print(" -", VIDEO[:-4] + "_overlay.mp4")
if OUT_CSV:
    print(" -", VIDEO[:-4] + ".csv")