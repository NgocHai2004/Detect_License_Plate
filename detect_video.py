from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# Load mô hình YOLO
model = YOLO("license_plate_detector.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Mở video
cap = cv2.VideoCapture("input/13168477_2560_1440_25fps.mp4")  # hoặc 0 nếu là webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Cắt ảnh biển số để OCR
            cropped = frame[y1:y2, x1:x2]
            ocr_result = reader.readtext(cropped)

            # Mặc định text là "UNKNOWN"
            text = "UNKNOWN"
            if ocr_result:
                text = ocr_result[0][1].upper()

            # Vẽ bounding box màu xanh
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Tạo nền đen để hiển thị chữ
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 0, 0), -1)  # nền đen

            # Ghi chữ màu trắng lên nền đen
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
    resized = cv2.resize(frame, (960, 540))  # hoặc (1280, 720) nếu bạn muốn lớn hơn
    cv2.imshow("License Plate Detection", resized)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
