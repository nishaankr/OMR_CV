import cv2
import os

# Point to ONE sample image and its label
image_path = "Train/images/IMG_1584_iter_0.jpg"   # change to any image name in your train folder
label_path = "Train/labels/IMG_1584_iter_0.txt"   # same name, .txt extension

# Load image
img = cv2.imread(image_path)
h, w = img.shape[:2]   # get actual pixel height and width

# Read labels and draw boxes
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    cls   = int(parts[0])
    cx    = float(parts[1]) * w    # convert normalized → pixel
    cy    = float(parts[2]) * h    # this is center_y in pixels
    bw    = float(parts[3]) * w
    bh    = float(parts[4]) * h

    # Draw rectangle
    x1 = int(cx - bw/2)
    y1 = int(cy - bh/2)
    x2 = int(cx + bw/2)
    y2 = int(cy + bh/2)

    color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # red=filled, green=empty
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Print center_y normalized value next to bubble
    cv2.putText(img, f"{float(parts[2]):.2f}", (x1, y1-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

# Save result
cv2.imwrite("visualized_labels.jpg", img)
print(f"Image size: {w} x {h}")
print(f"Saved to: visualized_labels.jpg")
print(f"Total labels: {len(lines)}")
