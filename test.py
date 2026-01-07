from ultralytics import YOLO

model = YOLO("models/best.engine", task="detect", verbose=True)
results = model.predict(source="2_720p.mp4", imgsz=640, save=False, conf=0.01, show=True, device='cuda:0', half=True)


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    confidences = boxes.conf.tolist()  # List of confidences per box
    print(min(confidences))    