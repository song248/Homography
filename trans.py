import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_A_path = "test2_yolov8x_result.jpg"  # 탐지 결과가 있는 이미지
image_B_path = "floor.jpg"  # 평면도 이미지
image_A = cv2.imread(image_A_path)
image_B = cv2.imread(image_B_path)

# A 이미지에서 수동으로 대응점 설정 (예제 좌표, 실제 이미지에 맞게 수정해야 함)
src_pts = np.array([
    [100, 200],  # A 이미지의 점1 (예제)
    [400, 300],  # A 이미지의 점2
    [150, 500],  # A 이미지의 점3
    [500, 600]   # A 이미지의 점4
], dtype=np.float32)

# B 이미지에서 대응하는 실제 평면도 좌표 설정
dst_pts = np.array([
    [50, 100],   # B 이미지에서의 점1
    [300, 200],  # B 이미지에서의 점2
    [80, 450],   # B 이미지에서의 점3
    [350, 550]   # B 이미지에서의 점4
], dtype=np.float32)

# Homography 행렬 계산
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# A 이미지에서 감지된 사람 중심 좌표 (예제 좌표, YOLO 결과에서 추출해야 함)
people_A = np.array([
    [250, 400],  # 사람 1
    [500, 500],  # 사람 2
    [700, 600]   # 사람 3
], dtype=np.float32)

# 좌표 변환 함수
def transform_points(H, points):
    transformed_points = []
    for point in points:
        p = np.array([point[0], point[1], 1], dtype=np.float32).T
        p_transformed = np.dot(H, p)
        p_transformed /= p_transformed[2]  # Homogeneous 좌표 변환
        transformed_points.append((int(p_transformed[0]), int(p_transformed[1])))
    return transformed_points

# 사람 좌표 변환
people_B = transform_points(H, people_A)

# 변환된 좌표를 평면도에 표시
for (x, y) in people_B:
    cv2.circle(image_B, (x, y), 10, (0, 0, 255), -1)  # 빨간 점

# 결과 이미지 저장
output_path = "floor_mapped.jpg"
cv2.imwrite(output_path, image_B)
print(f"Transformed people locations saved to: {output_path}")