from ultralytics import YOLO
import cv2

# === Load mô hình YOLOv8 nhận diện biển số ===
model = YOLO("license_plate_detector.pt")

# === Load ảnh đầu vào ===
img_path = "input/d.jpg"
image = cv2.imread(img_path)

# === Phát hiện biển số với YOLO ===
results = model(image)

# === Duyệt qua từng kết quả phát hiện ===
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # === Vẽ bounding box ===
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === Ghi chữ "CAR" thay cho kết quả OCR ===
        cv2.putText(image, "CAR", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

# === Hiển thị ảnh kết quả ===
cv2.imshow("Phát hiện biển số", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
