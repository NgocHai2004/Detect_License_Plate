from ultralytics import YOLO
import cv2
import os

# Tải mô hình YOLOv8 đã huấn luyện
file_path = r"best.pt"
model = YOLO(file_path)

# Mở video
file = r'demo.mp4'  # Đảm bảo đây là video
cap = cv2.VideoCapture(file)

# Kiểm tra nếu video không thể mở
if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

# Tạo thư mục gốc để lưu file .txt và ảnh đã đánh nhãn nếu chưa tồn tại
output_base_dir = 'output'
os.makedirs(output_base_dir, exist_ok=True)

# Đọc video và dự đoán trên từng khung
frame_count = 1
while cap.isOpened():
    ret, frame = cap.read()  # Đọc từng khung hình của video
    if not ret:
        print("Không còn khung hình để đọc hoặc lỗi khi đọc.")
        break

    # Thực hiện dự đoán trên khung hình
    results = model.predict(source=frame, show=False, conf=0.25)

    # Kiểm tra nếu có phát hiện đối tượng
    if results[0].boxes:  # Nếu có bounding box trong kết quả
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Lấy tọa độ bounding box
            confidences = result.boxes.conf.cpu().numpy()  # Lấy độ tin cậy
            class_ids = result.boxes.cls.cpu().numpy()  # Lấy nhãn (class ID)
            height, width, _ = frame.shape  # Kích thước khung hình

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if class_id == 0:  # Chỉ xử lý biển số xe
                    # Tạo thư mục theo nhãn biển số xe
                    emotion_folder = os.path.join(output_base_dir, f"license_plate")
                    os.makedirs(emotion_folder, exist_ok=True)

                    # Lưu file .txt cho từng biển số
                    with open(os.path.join(emotion_folder, f'frame_{frame_count}.txt'), 'w') as f:
                        x1, y1, x2, y2 = box
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                    # Vẽ kết quả dự đoán lên khung hình
                    annotated_frame = results[0].plot()
                    cv2.imwrite(os.path.join(emotion_folder, f'frame_{frame_count}.jpg'), annotated_frame)

        # Tăng số đếm khung hình được lưu
        frame_count += 1

        # Hiển thị khung hình đã được dự đoán
        cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
