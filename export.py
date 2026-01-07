from ultralytics import YOLO

model = YOLO("models/best.pt")
model.export(format="engine",
                            task="detect",
                            imgsz=736,
                            batch=16,
                            half=False,
                            nms=True,
                            dynamic=True,
                            int8=True,
                            data="dataset/data.yaml"
                        )