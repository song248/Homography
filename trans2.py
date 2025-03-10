import cv2
import numpy as np
import os

image_A_path = "test2_yolov8x_result.jpg" 
image_B_path = "floor.jpg"

image_A = cv2.imread(image_A_path)
image_B = cv2.imread(image_B_path)

# Green dot
lower_green = np.array([0, 250, 0], dtype=np.uint8)
upper_green = np.array([10, 255, 10], dtype=np.uint8)
mask = cv2.inRange(image_A, lower_green, upper_green)
people_A = np.column_stack(np.where(mask > 0))
people_A = [(x, y) for y, x in people_A]

src_pts = np.array([
    [150, 200],  # A 이미지의 기준점1 (좌상단 → 오른쪽으로 이동)
    [1150, 200], # A 이미지의 기준점2 (우상단)
    [150, 850],  # A 이미지의 기준점3 (좌하단 → 아래쪽으로 더 이동)
    [1150, 850]  # A 이미지의 기준점4 (우하단)
], dtype=np.float32)

dst_pts = np.array([
    [200, 200],   # B 이미지의 기준점1 (좌상단 → 오른쪽으로 이동)
    [1000, 200],  # B 이미지의 기준점2 (우상단 → 더 오른쪽)
    [220, 920],   # B 이미지의 기준점3 (좌하단 → 더 아래로 이동)
    [1020, 920]   # B 이미지의 기준점4 (우하단 → 더 아래로 이동)
], dtype=np.float32)

# Homography 행렬 계산
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

def transform_points(H, points):
    transformed_points = []
    for point in points:
        p = np.array([point[0], point[1], 1], dtype=np.float32).T
        p_transformed = np.dot(H, p)
        p_transformed /= p_transformed[2]
        transformed_points.append((int(p_transformed[0]), int(p_transformed[1])))
    return transformed_points

people_B = transform_points(H, people_A)
for (x, y) in people_B:
    cv2.circle(image_B, (x, y), 5, (0, 0, 255), -1)  # 빨간 점

output_path = "floor_mapped2.jpg"
cv2.imwrite(output_path, image_B)
print(f"Transformed image saved to: {output_path}")
