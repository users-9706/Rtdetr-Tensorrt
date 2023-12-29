import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import math
import random
random.seed(3)
CLASS_COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
def alloc_buf_N(engine, data):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    data_type = []
    for binding in engine:
        if engine.binding_is_input(binding):
            size = data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            data_type.append(dtype)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, data_type[0])
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
def do_inference_v2(context, inputs, bindings, outputs, stream, data):
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    context.set_binding_shape(0, data.shape)
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    stream.synchronize()
    return [out.host for out in outputs]
trt_logger = trt.Logger(trt.Logger.INFO)
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
def bbox_cxcywh_to_xyxy(x):
    bbox = np.zeros_like(x)
    bbox[..., :2] = x[..., :2] - 0.5 * x[..., 2:]
    bbox[..., 2:] = x[..., :2] + 0.5 * x[..., 2:]
    return bbox
if __name__ == '__main__':
    class_num = 80
    keep_boxes = 300
    conf_thres = 0.45
    image = cv2.imread("bus.jpg")
    image_h, image_w = image.shape[:2]
    ratio_h = 640 / image_h
    ratio_w = 640 / image_w
    img = cv2.resize(image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2)
    img = img[:, :, ::-1] / 255.0
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img[np.newaxis], dtype=np.float32)
    engine = load_engine("rtdetr-l.engine")
    context = engine.create_execution_context()
    inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(engine, img)
    inputs_alloc_buf[0].host = img.reshape(-1)
    net_output = do_inference_v2(context, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf, stream_alloc_buf,
                                 img)
    net_output = net_output[0].reshape(keep_boxes, -1)
    boxes = net_output[:, :4]
    scores = net_output[:, 4:]
    boxes = bbox_cxcywh_to_xyxy(boxes)
    _max = scores.max(-1)
    _mask = _max > conf_thres
    boxes, scores = boxes[_mask], scores[_mask]
    boxes = boxes * np.array([640, 640, 640, 640], dtype=np.float32)
    labels = scores.argmax(-1)
    scores = scores.max(-1)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 = math.floor(min(max(1, x1 / ratio_w), image_w - 1))
        y1 = math.floor(min(max(1, y1 / ratio_h), image_h - 1))
        x2 = math.ceil(min(max(1, x2 / ratio_w), image_w - 1))
        y2 = math.ceil(min(max(1, y2 / ratio_h), image_h - 1))
        cv2.rectangle(image, (x1, y1), (x2, y2), CLASS_COLORS[label], 2)
        cv2.putText(image, f'{CLASS_NAMES[label]} : {score:.2f}',
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
    cv2.imshow("result", image)
    cv2.waitKey(0)
