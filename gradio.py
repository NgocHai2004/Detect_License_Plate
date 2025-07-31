import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import csv
from datetime import datetime
import os

# === Load YOLOv8 & EasyOCR ===
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'], gpu=False)
CSV_PATH = "plates.csv"

# === Định dạng biển số kiểu Việt Nam ===
def format_plate_vn(text):
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    match = re.match(r"(\d{2})([A-Z]{1,2})(\d{4,5})", cleaned)
    if match:
        return "".join(match.groups())
    return cleaned if len(cleaned) >= 6 else None

# === CSV ===
def save_plate_to_csv(plate):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([plate, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def remove_plate_from_csv(plate):
    if not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, "r") as f:
        rows = list(csv.reader(f))
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            if row and row[0] != plate:
                writer.writerow(row)

def load_plates_from_csv():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r") as f:
        return [row[0] for row in csv.reader(f) if row]

# === Xử lý ảnh ===
def detect_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    plates = []
    stored_plates = load_plates_from_csv()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        plate_img = image_rgb[y1:y2, x1:x2]
        result = reader.readtext(plate_img)
        text = "".join([r[1] for r in result])
        plate = format_plate_vn(text)
        if plate:
            plates.append(plate)
            if plate not in stored_plates:
                save_plate_to_csv(plate)
            else:
                remove_plate_from_csv(plate)

            # Vẽ biển số lên ảnh
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_rgb, plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image_rgb, "\n".join(plates) if plates else "Không phát hiện biển số nào."

# === Xử lý video ===
def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame

    cap.release()

    if last_frame is None:
        return None, "Không thể đọc frame cuối video."

    return detect_from_image(last_frame)

# === Giao diện Gradio ===
with gr.Blocks(css="body {background-color: black; color: white;}") as demo:
    gr.Markdown("<h1 style='color:white;'>🚘 Nhận diện biển số xe bằng YOLOv8 + EasyOCR</h1>")

    with gr.Tab("📷 Ảnh"):
        image_input = gr.Image(type="numpy", label="Chọn ảnh biển số")
        image_output = gr.Image(label="Ảnh đã nhận diện")
        image_result = gr.Textbox(label="📄 Biển số phát hiện")

        image_btn = gr.Button("🔍 Nhận diện từ ảnh")
        image_btn.click(fn=detect_from_image, inputs=image_input, outputs=[image_output, image_result])

    with gr.Tab("🎥 Video"):
        video_input = gr.Video(label="Tải lên video")
        video_output = gr.Image(label="Khung hình cuối đã xử lý")
        video_result = gr.Textbox(label="📄 Biển số phát hiện")

        video_btn = gr.Button("▶️ Nhận diện từ video")
        video_btn.click(fn=detect_from_video, inputs=video_input, outputs=[video_output, video_result])

demo.launch()
