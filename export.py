from ultralytics import YOLO
model = YOLO("rtdetr-l.pt")  
model.export(format="engine",opset=16)
