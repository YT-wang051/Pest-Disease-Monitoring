from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")
results = model("D:/YOLOv11/ultralytics-8.3.39/ultralytics/assets",conf=0.25,save=True)

