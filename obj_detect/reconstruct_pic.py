import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ----------------- 重建圖片 -----------------
# 讀取 CSV 檔案，CSV 格式應包含欄位: u, v, R, G, B
df = pd.read_csv('pointcloud_20250225_030852.csv')

# 設定圖片尺寸，假設 u 對應列（寬度）、v 對應行（高度）
width = 640
height = 480

# 初始化圖片陣列 (uint8 格式)
image = np.zeros((height, width, 3), dtype=np.uint8)

# 將 CSV 中每個像素的 RGB 資料填入圖片陣列
for idx, row in df.iterrows():
    u = int(row['u'])  # 水平（列）位置
    v = int(row['v'])  # 垂直（行）位置
    R = int(row['R'])
    G = int(row['G'])
    B = int(row['B'])
    image[v, u] = [R, G, B]

# ----------------- LVIS 物件偵測（Detectron2） -----------------
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

cfg = get_cfg()
# 使用 LVIS 模型的配置檔，此模型基於 LVISv0.5 訓練
config_path = "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_path))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 設定信心值閥值
cfg.MODEL.DEVICE = "cpu"

metadata = MetadataCatalog.get("lvis_v0.5_val")
predictor = DefaultPredictor(cfg)
outputs = predictor(image)

# ----------------- 解析偵測結果 -----------------
# 將 outputs 轉成 CPU 上的 instances
instances = outputs["instances"].to("cpu")
boxes = instances.pred_boxes.tensor.numpy()  # 每個邊界框格式 [x1, y1, x2, y2]
classes = instances.pred_classes.numpy()       # 類別索引
scores = instances.scores.numpy()                # 信心分數

# 利用 metadata 將類別索引轉換成文字標籤
detection_labels = [metadata.thing_classes[c] for c in classes]

# 計算每個偵測結果的面積，並加入 DataFrame
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
detection_results = pd.DataFrame({
    "label": detection_labels,
    "score": scores,
    "x1": boxes[:, 0],
    "y1": boxes[:, 1],
    "x2": boxes[:, 2],
    "y2": boxes[:, 3],
    "area": areas
})

# 依照面積由小到大排序，讓最小的優先標記
detection_results = detection_results.sort_values(by="area", ascending=True)
print("偵測結果（依面積排序）：")
print(detection_results)

# ----------------- 更新原始 CSV -----------------
# 在原始 DataFrame 中加入欄位，預設為空值或 NaN
df["object_label"] = ""
df["bbox_x1"] = np.nan
df["bbox_y1"] = np.nan
df["bbox_x2"] = np.nan
df["bbox_y2"] = np.nan

# 依照排序結果進行更新
# 只對那些尚未被標記的像素進行更新
for idx, det in detection_results.iterrows():
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    label = det["label"]
    
    # 使用條件過濾：像素位於偵測邊界框內，且 object_label 欄位仍為空
    mask = (df["u"] >= x1) & (df["u"] <= x2) & (df["v"] >= y1) & (df["v"] <= y2) & (df["object_label"] == "")
    df.loc[mask, "object_label"] = label
    df.loc[mask, "bbox_x1"] = round(x1, 2)
    df.loc[mask, "bbox_y1"] = round(y1, 2)
    df.loc[mask, "bbox_x2"] = round(x2, 2)
    df.loc[mask, "bbox_y2"] = round(y2, 2)

# 存回 CSV 檔案 (例如 "pointcloud_with_objects.csv")
df.to_csv("pointcloud_with_objects.csv", index=False)
print("更新後的 CSV 檔案已儲存到 pointcloud_with_objects.csv")

# ----------------- 顯示視覺化結果 -----------------
v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
out = v.draw_instance_predictions(instances)
result_image = out.get_image()[:, :, ::-1]  # 轉回 BGR

plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
