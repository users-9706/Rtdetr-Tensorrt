    Environment: win11, pytorch2.0, cuda11.8, cudnn8.6, tensorrt8.6, Opencv4.8
    1. export onnx
        from ultralytics import YOLO 
        model = YOLO("rtdetr-l.pt") 
        model.export(format="onnx", opset=16)
    2. Generate engine
       trtexec.exe --onnx=rtdetr-l.onnx --saveEngine=rtdetr-l.engine
