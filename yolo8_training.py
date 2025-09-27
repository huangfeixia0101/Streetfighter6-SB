# 安装 YOLOv8
pip install ultralytics

# 一键训练（100 epoch，自动划分 train/val）
yolo train task=detect data=yolo_dataset names=kevin,ryu epochs=100 imgsz=640 batch=16