from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8x.pt")
image_path = "test2.jpg"
image = cv2.imread(image_path)

results = model(image, conf=0.09)

people_centers = []
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            people_centers.append((center_x, center_y))

            cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

print("Detected people coordinates:", people_centers)
output_image_path = image_path.replace('.jpg', '_')+'yolov8x_result.jpg'
cv2.imwrite(output_image_path, image)
print(f"Detection result saved as: {output_image_path}")
