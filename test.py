from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/omr_phan1_new/weights/best.pt')

    metrics = model.val(
        data='dataset.yaml',
        split='test',
        name='omr_test_results'
    )

    print("\n=== TEST SET RESULTS ===")
    print(f"mAP50:      {metrics.box.map50:.3f}")
    print(f"mAP50-95:   {metrics.box.map:.3f}")
    print(f"Precision:  {metrics.box.p.mean():.3f}")
    print(f"Recall:     {metrics.box.r.mean():.3f}")
