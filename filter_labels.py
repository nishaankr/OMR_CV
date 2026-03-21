import os
import shutil

# ── SET THESE after checking Step 1 and 2 output ──────────────
Y_MIN = 0.35    # just below the first Phần I bubble row
Y_MAX = 0.55    # just above where Phần II starts
# ──────────────────────────────────────────────────────────────

splits = ['Train', 'Valid', 'Test']

for split in splits:
    input_label_dir  = f"{split}/labels"
    output_label_dir = f"{split}/labels_phan1"   # new filtered folder

    os.makedirs(output_label_dir, exist_ok=True)

    total_files    = 0
    total_kept     = 0
    total_removed  = 0

    for filename in os.listdir(input_label_dir):
        if not filename.endswith('.txt'):
            continue

        input_path  = os.path.join(input_label_dir, filename)
        output_path = os.path.join(output_label_dir, filename)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        # Keep only labels where center_y is inside Phần I range
        filtered = [
            line for line in lines
            if line.strip() and Y_MIN <= float(line.strip().split()[2]) <= Y_MAX
        ]

        # Write filtered labels (even if empty — YOLO expects a file per image)
        with open(output_path, 'w') as f:
            f.writelines(filtered)

        total_files   += 1
        total_kept    += len(filtered)
        total_removed += len(lines) - len(filtered)

    print(f"{split}: {total_files} files | kept {total_kept} labels | removed {total_removed} labels")

print("\nDone! New label folders created: Train/labels_phan1, Valid/labels_phan1, Test/labels_phan1")
