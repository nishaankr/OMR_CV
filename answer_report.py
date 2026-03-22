import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

model    = YOLO('runs/detect/omr_phan1_new/weights/best.pt')
img_dir  = 'Datasets/YOLO_OMR_Dataset/test/images/'
out_dir  = 'runs/omr_extraction'
os.makedirs(out_dir, exist_ok=True)

COLUMNS     = ['A', 'B', 'C', 'D']
Y_TOLERANCE = 15
CONF        = 0.3

def extract_answers(img_path):
    img_full = cv2.imread(img_path)
    h, w     = img_full.shape[:2]

    y1_crop = int(h * 0.35)
    y2_crop = int(h * 0.55)
    img     = img_full[y1_crop:y2_crop, 0:w]

    results = model.predict(source=img, conf=CONF, verbose=False)

    bubbles = []
    for box in results[0].boxes:
        cls             = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx              = (x1 + x2) / 2
        cy              = (y1 + y2) / 2
        bubbles.append({'cls': cls, 'cx': cx, 'cy': cy})

    if len(bubbles) < 4:
        return {}, len(bubbles)

    cx_values        = sorted([b['cx'] for b in bubbles])
    gaps             = [(cx_values[i+1] - cx_values[i], i) for i in range(len(cx_values)-1)]
    gaps.sort(reverse=True)
    boundary_indices = sorted([gaps[i][1] for i in range(3)])
    boundaries       = [cx_values[i] for i in boundary_indices]

    def get_group(cx):
        for i, b in enumerate(boundaries):
            if cx <= b:
                return i
        return 3

    x_groups = {0: [], 1: [], 2: [], 3: []}
    for bubble in bubbles:
        x_groups[get_group(bubble['cx'])].append(bubble)

    answers = {}
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
            answer = '—'
            if len(row) == 4:
                row.sort(key=lambda b: b['cx'])
                for col_idx, bubble in enumerate(row):
                    if bubble['cls'] == 0:
                        answer = COLUMNS[col_idx]
                        break
            answers[q_num] = answer

    return answers, len(bubbles)


if __name__ == '__main__':
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    all_results = []

    print(f"Processing {len(image_files)} images...\n")

    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(img_dir, filename)
        answers, bubble_count = extract_answers(img_path)

        answered = sum(1 for v in answers.values() if v != '—')

        record = {
            'image'       : filename,
            'bubble_count': bubble_count,
            'answered'    : answered,
            'blank'       : 40 - answered,
            'answers'     : {f'Q{k}': v for k, v in answers.items()}
        }
        all_results.append(record)

        print(f"[{idx:>4}/{len(image_files)}] {filename:<35} bubbles={bubble_count:<5} answered={answered}/40")

    # Save as json
    json_path = os.path.join(out_dir, 'all_answers.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! {len(image_files)} images processed.")
    print(f"Saved {json_path}")
