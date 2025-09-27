import cv2, numpy as np, csv, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO   = r"C:\Users\huang\Videos\SF6\SF6.mp4"
CSV_OUT = "best_hp_roi.csv"
# HSV 阈值保持原样
def hp_ratio(roi):
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (165, 250, 115), (170, 255, 210)) |
            cv2.inRange(hsv, (100, 200, 130), (115, 225, 190)) |
            cv2.inRange(hsv, (25,  50, 240), (35, 150, 255)))
    col_flag = np.sum(mask > 0, axis=0) >= max(1, roi.shape[0] // 10)
    return np.count_nonzero(col_flag) / len(col_flag)

# 每个线程独立打开视频，避免锁
def eval_one_cfg(args):
    x, y, w, off = args
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    best_l = best_r = 0.0
    for idx in range(50):               # 只读前 50 帧
        ret, frame = cap.read()
        if not ret:
            break
        left  = hp_ratio(frame[y:y+24, x:x+w])
        right = hp_ratio(frame[y:y+24, x+off:x+off+w])
        best_l, best_r = max(best_l, left), max(best_r, right)
        # 只要有一帧同时≥98 % 就立即算分并返回，省掉后面帧
        if left >= 0.98 and right >= 0.98:
            break
    cap.release()
    score = 1 - abs(best_l - 1) - abs(best_r - 1)
    return (x, y, w, 24, off, score, best_l, best_r)

# 生成任务
tasks = [(x, y, w, off)
         for x in range(185-5, 185+6)
         for y in range(68-5, 68+6)
         for w in range(665-5, 665+6)
         for off in range(882-5, 882+6)]

candidates = []
print('=== 多线程扫描（100帧即停）===')
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    futures = {pool.submit(eval_one_cfg, t): t for t in tasks}
    for f in as_completed(futures):
        x, y, w, h, off, score, l, r = f.result()
        if l >= 0.97 and r >= 0.97:          # 只保留双边≥98 %
            candidates.append((x, y, w, h, off, score, l, r))

# 按 score 降序，取最佳
candidates.sort(key=lambda x: x[5], reverse=True)
if candidates:
    best = candidates[0]
    print('\n=== 前 100 帧内 双边≥97 % 的最佳组 ===')
    print(f'ROI_LEFT  = {best[:4]}')
    print(f'ROI_RIGHT = ({best[0]+best[4]}, {best[1]}, {best[2]}, {best[3]})')
    print(f'偏移量 = {best[4]}  |  score={best[5]:.4f}  (L={best[6]:.4f}, R={best[7]:.4f})')

    # 写 CSV
    with open(CSV_OUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y','w','h','offset','score','left','right'])
        writer.writerows(candidates)
    print(f'\n已写入 {len(candidates)} 组候选到 {CSV_OUT}')
else:
    print('前 100 帧内未出现双边同时≥97 % 的情况')