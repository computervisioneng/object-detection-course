import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')


def detect_objects(image):
    results = model(image)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        detections.append([int(x1), int(y1), int(x2), int(y2), round(score, 3),
                           results.names[int(class_id)]])

    return detections


for detection in detect_objects('./image.jpg'):
    print(detection)
