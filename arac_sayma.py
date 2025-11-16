import cv2
from PIL import Image
from ultralytics import YOLO
import argparse
import numpy as np
import os

# ------------------ Argümanlar ------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    type=str,
    default="/home/selmandev/Desktop/Arac_Sayma_Sistemi/deneme_videosu.mp4",
    help="Kamera index (örn. 0) veya video yolu"
)

parser.add_argument("--algo", type=str, default="MOG2", choices=["MOG2", "KNN"], help="Arka plan çıkarma")
parser.add_argument("--width", type=int, default=1280, help="Kamera genişliği")
parser.add_argument("--height", type=int, default=720, help="Kamera yüksekliği")
args = parser.parse_args()

# ------------------ Model & Sabitler ------------------
kernel = np.ones((3,3), np.uint8)
model = YOLO("yolov8l.pt")

# ------------------ Yardımcılar ------------------
def build_video_source(source: str, width: int, height: int) -> cv2.VideoCapture:
    #if else dongusu kamera kaynaginin int veya str(path) olmasına gore duzenlenmıstır
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise ValueError("Kamera açılamadı.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video bulunamadı: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError("Video açılamadı.")
    return cap

def build_bg_subtractor(method: str):
    if method == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=70, detectShadows=False)
    else:
        return cv2.createBackgroundSubtractorKNN()

def detect_objects(image):
    """BGR görüntüde YOLO tespiti yapar ve car/bus/truck döndürür."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    results = model.predict(image_pil, verbose=False)[0]

    detections = []
    for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
        label = results.names[int(class_id)]
        if score >= 0.3 and label in ["car", "bus", "truck"]:
            detections.append([int(x1), int(y1), int(x2), int(y2), round(float(score), 3), label])
    return detections

# ------------------ Ana Döngü ------------------
def open_camera():
    cap = build_video_source(args.source, args.width, args.height)
    backSub = build_bg_subtractor(args.algo)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FG maske
            fgMask = backSub.apply(frame)
            fgMask_clean = fgMask.copy()  # hareket yoksa da gösterimde kullanmak için

            # Yeterli hareket varsa kontur analizi
            if cv2.countNonZero(fgMask) > 5000:  
                #BURASI PROBLEMLİ BURAYA BAKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
                fgMask_clean = cv2.morphologyEx(fgMask,cv2.MORPH_OPEN,kernel)
                fgMask_clean = cv2.dilate(fgMask,kernel,iterations=1)
                _, thresh = cv2.threshold(fgMask_clean, 200, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1000:
                        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

            # YOLO tespiti
            detections = detect_objects(frame)
            for x1, y1, x2, y2, score, label in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Göster
            cv2.imshow("Kamera", frame)
            cv2.imshow("FG Mask", fgMask)
            cv2.imshow("Clean Mask", fgMask_clean)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()
