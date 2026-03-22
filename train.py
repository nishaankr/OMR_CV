import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

if __name__ == '__main__':  
    torch.cuda.empty_cache()  
    
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='dataset.yaml',
        epochs=30,
        imgsz=640,
        batch=6,           # ← reduced from 8 to 6
        device=0,
        workers=2,
        cache=False,
        mosaic=0.5,        # ← reduce mosaic from 1.0 to 0.5 (less memory spikes)
        name='omr_phan1_new'
    )
