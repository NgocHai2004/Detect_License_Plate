from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import csv
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Đảm bảo index.html nằm trong /templates

model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'], gpu=False)

CSV_PATH = "detect.csv"

# === Load plates đã vào ===
def load_detected_plates():
    plates = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader_csv = csv.reader(f)
            next(reader_csv, None)
            for row in reader_csv:
                if row:
                    plates[row[0]] = row[1]
    return plates

# === Ghi file CSV mới sau khi xóa 1 plate ===
def write_detected_plates(plates_dict):
    with open(CSV_PATH, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "datetime"])
        for plate, dt in plates_dict.items():
            writer.writerow([plate, dt])

# === Giải mã base64 ảnh ===
def decode_base64_image(data_uri):
    if "," in data_uri:
        data = data_uri.split(",")[1]
    else:
        data = data_uri
    image_data = base64.b64decode(data)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# === Hàm nhận diện và vẽ box ===
def detect_plate(image):
    results = model(image)
    detections = []
    plates_dict = load_detected_plates()

    # Clone ảnh gốc để vẽ box
    image_with_boxes = image.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            ocr_result = reader.readtext(crop)

            if ocr_result:
                text_parts = [item[1].upper().strip() for item in ocr_result]
                confidences = [item[2] for item in ocr_result]
                text = " ".join(text_parts)
                conf = float(np.mean(confidences))

                # Kiểm tra trạng thái vào/ra
                if text in plates_dict:
                    status = "exit"
                    del plates_dict[text]
                    write_detected_plates(plates_dict)
                else:
                    status = "enter"
                    plates_dict[text] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    write_detected_plates(plates_dict)

                # Vẽ bounding box + text
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Tính kích thước của text
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

                # Tọa độ khung nền chữ
                rect_x1 = x1
                rect_y1 = y1 - text_height - 10
                rect_x2 = x1 + text_width
                rect_y2 = y1

                # Vẽ nền đen phía sau chữ
                cv2.rectangle(image_with_boxes, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

                # Vẽ chữ màu trắng lên trên
                cv2.putText(image_with_boxes, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


                detections.append({
                    "text": text,
                    "confidence": conf,
                    "status": status
                })

    # Encode ảnh có box sang base64
    _, buffer = cv2.imencode('.jpg', image_with_boxes)
    image_with_box_b64 = base64.b64encode(buffer).decode('utf-8')

    return detections, image_with_box_b64

# === API ảnh ===
@app.route("/detect_image", methods=["POST"])
def detect_image():
    try:
        data = request.get_json()
        image_b64 = data.get("image")
        image = decode_base64_image(image_b64)
        detections, image_with_box_b64 = detect_plate(image)

        return jsonify({
            "success": True,
            "detections": detections,
            "total_plates": len(detections),
            "image_with_box": image_with_box_b64  # 👈 thêm dòng này
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route("/reset_csv", methods=["POST"])
def reset_csv():
    try:
        with open(CSV_PATH, mode="w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["plate", "datetime"])  # Ghi lại tiêu đề cột
        return jsonify({"success": True, "message": "File detect.csv đã được reset!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# === API video frame ===
@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    try:
        data = request.get_json()
        image_b64 = data.get("image")
        video_time = data.get("video_time")
        image = decode_base64_image(image_b64)
        detections, image_with_box_b64 = detect_plate(image)

        return jsonify({
            "success": True,
            "detections": detections,
            "video_time": video_time,
            "total_plates": len(detections),
            "image_with_box": image_with_box_b64  # 👈 thêm dòng này
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# === Tạo file CSV nếu chưa có ===
if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["plate", "datetime"])
    app.run(debug=True)