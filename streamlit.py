import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import tempfile
import os
import csv

# === Load mô hình YOLOv8 và EasyOCR ===
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'], gpu=False)

# === Đường dẫn file CSV lưu biển số ===
CSV_PATH = "plates.csv"

# === Xử lý file CSV ===
def load_plates_from_csv():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return set(row[0] for row in reader if row)

def save_plate_to_csv(plate):
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([plate])

def remove_plate_from_csv(plate):
    if not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, 'r', newline='', encoding='utf-8') as f:
        rows = [row for row in csv.reader(f) if row and row[0] != plate]
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# === Hàm nhận diện biển số ===
def detect_license_plate(image):
    results = model(image)
    plates = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image[y1:y2, x1:x2]

            # Tiền xử lý
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            ocr_results = reader.readtext(thresh)
            ocr_results_sorted = sorted(ocr_results, key=lambda r: np.mean([pt[1] for pt in r[0]]))
            plate_texts = [text.strip() for (bbox, text, conf) in ocr_results_sorted if len(text.strip()) >= 3]
            plate_text = '\n'.join(plate_texts) if plate_texts else "UNKNOWN"

            # Vẽ box và nền đen + chữ xanh
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX

            for i, line in enumerate(plate_text.split('\n')):
                text_size = cv2.getTextSize(line, font, 0.8, 2)[0]
                y_text = y1 - 10 - (len(plate_texts)-1 - i)*30 if y1 > 60 else y2 + 10 + i*30
                cv2.rectangle(image, (x1, y_text - 25), (x1 + text_size[0] + 10, y_text + 5), (0, 0, 0), -1)
                cv2.putText(image, line, (x1 + 5, y_text), font, 0.8, (0, 255, 0), 2)

            plates.append(plate_text)

    return image, plates

# === Giao diện chính ===
st.title("🚘 Hệ thống nhận diện biển số xe")
option = st.radio("Chọn chế độ xử lý:", ["Ảnh", "Video"])

# === Ảnh ===
if option == "Ảnh":
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (1280, 720))

        processed_image, plate_texts = detect_license_plate(image)
        st.image(processed_image[:, :, ::-1], caption="Ảnh đã xử lý", channels="RGB")

        st.subheader("📄 Kết quả biển số:")
        stored_plates = load_plates_from_csv()

        if plate_texts:
            for i, plate in enumerate(plate_texts):
                st.success(f"Biển số {i+1}: \n{plate}")
                if plate not in stored_plates:
                    save_plate_to_csv(plate)
                    st.info(f"✅ Đã lưu biển số mới: {plate}")
                else:
                    remove_plate_from_csv(plate)
                    st.warning(f"⚠️ Biển số {plate} đã tồn tại, đã xóa khỏi CSV.")
        else:
            st.warning("Không phát hiện được biển số nào.")

# === Video ===
elif option == "Video":
    uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if 'pause' not in st.session_state:
            st.session_state.pause = False
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = None

        pause_btn = st.button("⏸️ Dừng" if not st.session_state.pause else "▶️ Tiếp tục")
        if pause_btn:
            st.session_state.pause = not st.session_state.pause

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        all_plates = set()

        while cap.isOpened():
            if st.session_state.pause:
                stframe.info("⏸️ Video đang tạm dừng...")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            st.session_state.last_frame = frame.copy()
            processed_frame, plate_texts = detect_license_plate(frame)

            for plate in plate_texts:
                all_plates.add(plate)

            stframe.image(processed_frame[:, :, ::-1], channels="RGB", use_column_width=True)

        cap.release()

        if not st.session_state.pause:
            st.success("✅ Video xử lý xong.")

        # Hiển thị khung hình cuối cùng khi dừng
        if st.session_state.pause and st.session_state.last_frame is not None:
            st.subheader("🖼️ Nhận diện khung hình hiện tại:")
            last_img, last_plates = detect_license_plate(st.session_state.last_frame.copy())
            st.image(last_img[:, :, ::-1], channels="RGB")

            if last_plates:
                for i, plate in enumerate(last_plates):
                    st.success(f"Biển số {i+1}: \n{plate}")
            else:
                st.warning("Không phát hiện được biển số nào ở khung hình hiện tại.")

        # Hiển thị và xử lý lưu/xóa biển số
        st.subheader("📄 Các biển số nhận được từ video:")
        stored_plates = load_plates_from_csv()

        if all_plates:
            for i, plate in enumerate(all_plates):
                st.success(f"Biển số {i+1}: \n{plate}")
                if plate not in stored_plates:
                    save_plate_to_csv(plate)
                    st.info(f"✅ Đã lưu biển số mới: {plate}")
                else:
                    remove_plate_from_csv(plate)
                    st.warning(f"⚠️ Biển số {plate} đã tồn tại, đã xóa khỏi CSV.")
        else:
            st.warning("Không phát hiện được biển số nào trong video.")