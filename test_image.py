from ultralytics import YOLO
import cv2
import os

# Tải mô hình YOLOv8 đã huấn luyện
file_path = r"best.pt"
model = YOLO(file_path)

# Thư mục chứa ảnh đầu vào
input_folder = 'images'
output_base_dir = 'output'
os.makedirs(output_base_dir, exist_ok=True)

# Lấy danh sách file ảnh trong thư mục
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Duyệt qua từng ảnh và dự đoán
for idx, image_name in enumerate(image_files, start=1):
    image_path = os.path.join(input_folder, image_name)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Không thể đọc ảnh: {image_name}")
        continue

    # Thực hiện dự đoán
    results = model.predict(source=frame, show=False, conf=0.25)

    # Kiểm tra nếu có phát hiện đối tượng
    if results[0].boxes:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            height, width, _ = frame.shape

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if class_id == 0:  # Chỉ xử lý class id = 0
                    # Tạo thư mục cho nhãn
                    emotion_folder = os.path.join(output_base_dir, "license_plate")
                    os.makedirs(emotion_folder, exist_ok=True)

                    # Lưu file .txt
                    with open(os.path.join(emotion_folder, f'image_{idx}.txt'), 'w') as f:
                        x1, y1, x2, y2 = box
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                    # Vẽ kết quả dự đoán lên ảnh
                    annotated_frame = results[0].plot()
                    cv2.imwrite(os.path.join(emotion_folder, f'image_{idx}.jpg'), annotated_frame)

    print(f"Xử lý xong ảnh: {image_name}")

print("Xong hết ảnh.")
