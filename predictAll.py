import cv2
import os
from ultralytics import YOLO

model = YOLO('runs/detect/omr_phan1_new/weights/best.pt')

input_dir  = 'Datasets/YOLO_OMR_Dataset/test/images/'

output_dir = 'runs/detect/omr_all_predict'
os.makedirs(output_dir, exist_ok=True)

colors = {0: (0, 0, 255), 1: (0, 200, 0)}

image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

for filename in image_files:
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    results = model.predict(source=img_path, conf=0.5, verbose=False)

    for box in results[0].boxes:
        cls  = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 1)
        if cls == 0:
            cv2.putText(img, f"F{conf:.2f}", (x1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, colors[cls], 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, filename), img)

print(f"Done! {len(image_files)} images saved to {output_dir}")
