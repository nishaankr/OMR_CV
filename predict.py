import cv2
import os
from ultralytics import YOLO

model = YOLO('runs/detect/omr_phan1_new/weights/best.pt')

img_path = 'Datasets/YOLO_OMR_Dataset/test/images/IMG_1584_iter_24.jpg'
img = cv2.imread(img_path)

results = model.predict(source=img_path, conf=0.5, verbose=False)

colors = {0: (0, 0, 255), 1: (0, 200, 0)}

for box in results[0].boxes:
    cls  = int(box.cls)
    conf = float(box.conf)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 1)
    if cls == 0:
        cv2.putText(img, f"F{conf:.2f}", (x1, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, colors[cls], 1, cv2.LINE_AA)


output_dir = 'runs/detect/omr_single_predict'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'IMG_1584_iter_24_predicted.jpg')
cv2.imwrite(output_path, img)
print(f"Saved to: {output_path}")
