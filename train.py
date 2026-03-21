import torch
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    workers=4,        
    name='omr_phan1_640_30_8'
)
