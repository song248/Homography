import cv2

# 동영상 파일 경로
video_path = "test_video.mp4"
output_image_path = "first_frame.jpg"

# OpenCV로 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("⚠️ 비디오 파일을 열 수 없습니다. (코덱 미지원, 파일 손상 가능성)")
else:
    print("✅ 비디오 파일이 정상적으로 열렸습니다.")

    # 첫 번째 프레임 확인
    ret, frame = cap.read()
    if ret:
        print("✅ 첫 번째 프레임을 정상적으로 읽었습니다.")

        # 프레임 크기 출력
        height, width, _ = frame.shape
        print(f"📏 프레임 크기: {width}x{height}")

        # 첫 번째 프레임 이미지 저장
        cv2.imwrite(output_image_path, frame)
        print(f"💾 첫 번째 프레임이 '{output_image_path}'로 저장되었습니다.")
    else:
        print("⚠️ 비디오 파일은 열렸지만, 프레임을 읽을 수 없습니다.")

cap.release()
