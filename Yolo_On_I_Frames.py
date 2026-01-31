import os
import csv
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# paths
FRAMES_DIR = "Frames"
OUTPUT_DIR = "Yolo_Output"
MODEL_PATH = "/home/utkarshs/Desktop/Cbir_Video/runs/detect/train2/weights/best.pt"

CONF_THRES = 0.25
IOU_THRES = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

for video_name in os.listdir(FRAMES_DIR):
    video_path = os.path.join(FRAMES_DIR, video_name)
    iframe_dir = os.path.join(video_path, "I")

    if not os.path.isdir(iframe_dir):
        continue

    print(f"\n▶ Running YOLO on I-frames of {video_name}")

    out_video_dir = os.path.join(OUTPUT_DIR, video_name)
    person_dir = os.path.join(out_video_dir, "persons")
    os.makedirs(person_dir, exist_ok=True)

    metadata_file = os.path.join(out_video_dir, "metadata.csv")

    with open(metadata_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video",
            "frame",
            "person_id",
            "x1", "y1", "x2", "y2",
            "confidence"
        ])

        for frame_name in tqdm(sorted(os.listdir(iframe_dir))):
            if not frame_name.endswith(".jpg"):
                continue

            frame_path = os.path.join(iframe_dir, frame_name)
            img = cv2.imread(frame_path)

            if img is None:
                continue

            results = model(img, conf=CONF_THRES, iou=IOU_THRES)[0]

            for pid, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # safety check
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_name = f"{frame_name[:-4]}_p{pid}.jpg"
                cv2.imwrite(os.path.join(person_dir, crop_name), crop)

                writer.writerow([
                    video_name,
                    frame_name,
                    pid,
                    x1, y1, x2, y2,
                    conf
                ])

    print(f"✅ Finished {video_name}")
