# OMR Answer Sheet Detection

<p align="center">
  <img src="assets/prediction_sample.jpg" width="480" alt="YOLOv8 Bubble Detection on OMR Sheet"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/mAP@0.5-0.943-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/F1_Score-0.90-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Inference-7.8ms-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

Automated bubble detection and answer extraction from Vietnamese MCQ (Phần I) answer sheets using a fine-tuned YOLOv8 model.

---

## Overview

This project trains YOLOv8 to detect and classify OMR bubbles as **filled** or **empty**, then extracts structured A/B/C/D answers for all 40 questions in the Phần I section. It was trained and evaluated on the [OMR Dataset by nghaanv](https://www.kaggle.com/datasets/nghaanv/omr-dataset) from Kaggle.

---

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | 0.943 |
| mAP@0.5:0.95 | 0.836 |
| Precision | 0.971 |
| Recall | 0.853 |
| F1 Score | 0.90 @ conf 0.213 |
| Inference Speed | 7.8 ms / image |

<p align="center">
  <img src="assets/BoxPR_curve.jpg" width="420" alt="Precision-Recall Curve"/>
  <img src="assets/BoxF1_curve.jpg" width="420" alt="F1-Confidence Curve"/>
</p>

---

## Setup

```bash
git clone https://github.com/nishaankr/omr-detection
cd omr-detection
python -m venv omr_env && source omr_env/bin/activate   # Windows: omr_env\Scripts\activate
pip install ultralytics opencv-python numpy torch torchvision
```

Place the dataset at:
```
Datasets/YOLO_OMR_Dataset/
    train/images/   train/labels/
    valid/images/   valid/labels/
    test/images/    test/labels/
```

---

## Usage

| Step | Script | Description |
|---|---|---|
| 1 | `python train.py` | Train YOLOv8 for 30 epochs |
| 2 | `python test.py` | Evaluate on 247 test images |
| 3 | `python predict.py` | Annotated prediction on single image |
| 4 | `python predictAll.py` | Batch prediction on all test images |
| 5 | `python grade.py` | Extract Q1–Q40 answers from one sheet |
| 6 | `python extract_all.py` | Batch extraction → `all_answers.json` |

---

## Output Format

```json
{
  "image": "IMG_1584_iter_24.jpg",
  "answered": 37,
  "blank": 3,
  "answers": { "Q1": "A", "Q2": "A", "Q3": "D", "Q4": "—" }
}
```

`—` indicates a blank (unanswered) question.

---

## Authors

| Name | GitHub |
|---|---|
| Nishaank Singh Rawat | [@nishaankr](https://github.com/nishaankr/) |
| Shanya Rai | [@shanya](#) <!-- Replace with Shanya's GitHub URL --> |

---

## Acknowledgements

Dataset sourced from Kaggle:  
**[OMR Dataset — nghaanv](https://www.kaggle.com/datasets/nghaanv/omr-dataset)**  
We gratefully acknowledge the dataset author for making this resource publicly available.

---

## License

This project is licensed under the [MIT License](LICENSE).
