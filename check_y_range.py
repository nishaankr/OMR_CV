label_path = "H:\GitRepos\OMR_CV\Datasets\YOLO_OMR_Dataset/Train/labels/IMG_1584_iter_0.txt"

with open(label_path, "r") as f:
    lines = f.readlines()

y_values = sorted([float(line.split()[2]) for line in lines if line.strip()])

print("Lowest 10 y-values (top of sheet):")
for y in y_values[:10]:
    print(f"  {y:.4f}")

print("\nMiddle y-values (around Phần I):")
mid = len(y_values) // 2
for y in y_values[mid-5:mid+5]:
    print(f"  {y:.4f}")

print("\nHighest 10 y-values (bottom of sheet):")
for y in y_values[-10:]:
    print(f"  {y:.4f}")
