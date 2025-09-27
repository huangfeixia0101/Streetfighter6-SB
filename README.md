step1: 检测血条
  标注固定位置（x,y,w,h）和颜色
  
step2: 检测人脸，使用yolo 
  用预训练 YOLO 人脸检测模型 → 从 SF6 视频里截取人脸框-https://github.com/lindevs/yolov8-face/releases
  保存每张人脸到文件夹（例如 faces/frame_0001_ryu.png）
  用人脸特征聚类 / 相似度分类 → 自动把相同角色的脸放一起（这里可以用 face_recognition / DeepFace / InsightFace）
  人工确认 / 重命名文件夹 → 给每个角色分配名字（Ken、Ryu、Chun-Li …）
  转成 YOLO 标注格式（txt + 图片）
  用 yolov8 重新训练，只识别 SF6 的人物脸
