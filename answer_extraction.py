import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('runs/detect/omr_phan1_new/weights/best.pt')

img_path = 'Datasets/YOLO_OMR_Dataset/test/images/IMG_1584_iter_24.jpg'
img_full = cv2.imread(img_path)
h, w     = img_full.shape[:2]

# cropping to only take required section
y1_crop = int(h * 0.35)
y2_crop = int(h * 0.55)
img     = img_full[y1_crop:y2_crop, 0:w]

results = model.predict(source=img, conf=0.3, verbose=False)

# Detection
bubbles = []
for box in results[0].boxes:
    cls             = int(box.cls)
    conf            = float(box.conf)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx              = (x1 + x2) / 2
    cy              = (y1 + y2) / 2
    bubbles.append({'cls': cls, 'conf': conf,
                    'cx': cx,  'cy': cy,
                    'x1': x1,  'y1': y1,
                    'x2': x2,  'y2': y2})

# group boundaries
cx_values        = sorted([b['cx'] for b in bubbles])
gaps             = [(cx_values[i+1] - cx_values[i], i) for i in range(len(cx_values)-1)]
gaps.sort(reverse=True)
boundary_indices = sorted([gaps[i][1] for i in range(3)])
boundaries       = [cx_values[i] for i in boundary_indices]

# Assigning bubbles
def get_group(cx, boundaries):
    for i, b in enumerate(boundaries):
        if cx <= b:
            return i
    return 3

x_groups = {0: [], 1: [], 2: [], 3: []}
for bubble in bubbles:
    x_groups[get_group(bubble['cx'], boundaries)].append(bubble)

# Extact Ans
Y_TOLERANCE = 15
COLUMNS     = ['A', 'B', 'C', 'D']
answers     = {}

for group_idx, group_bubbles in x_groups.items():
    q_start = group_idx * 10 + 1

    if not group_bubbles:
        continue

    group_bubbles.sort(key=lambda b: b['cy'])

    rows        = []
    current_row = [group_bubbles[0]]

    for bubble in group_bubbles[1:]:
        if abs(bubble['cy'] - current_row[0]['cy']) <= Y_TOLERANCE:
            current_row.append(bubble)
        else:
            rows.append(current_row)
            current_row = [bubble]
    if current_row:
        rows.append(current_row)

    for row_idx, row in enumerate(rows):
        if row_idx >= 10:
            break

        q_num  = q_start + row_idx
        answer = '—'    # blank by default

        if len(row) == 4:
            row.sort(key=lambda b: b['cx'])
            for col_idx, bubble in enumerate(row):
                if bubble['cls'] == 0:   # filled
                    answer = COLUMNS[col_idx]
                    break

        answers[q_num] = answer

# Resulkts
print(f"\n{'='*30}")
print(f"  OMR ANSWER EXTRACTION")
print(f"{'='*30}")
print(f"{'Q':<6} {'Answer'}")
print(f"{'-'*15}")

for q in sorted(answers.keys()):
    print(f"Q{q:<5} {answers[q]}")

# Summary
filled_count = sum(1 for v in answers.values() if v != '—')
blank_count  = sum(1 for v in answers.values() if v == '—')

print(f"\n{'='*30}")
print(f"Total Questions : 40")
print(f"Answered        : {filled_count}")
print(f"Left Blank      : {blank_count}")
