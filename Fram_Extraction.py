import os
import subprocess
import shutil
from tqdm import tqdm

VIDEO_DIR = "Videos"
OUTPUT_DIR = "Frames"
TMP_DIR = "tmp_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

def get_frame_types(video_path):
    cmd = [
        "ffprobe",
        "-select_streams", "v",
        "-show_frames",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip().splitlines()

def extract_all_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        os.path.join(out_dir, "%06d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for video_name in os.listdir(VIDEO_DIR):
    if not video_name.endswith(".mp4"):
        continue

    print(f"\n▶ Processing {video_name}")

    video_path = os.path.join(VIDEO_DIR, video_name)
    video_id = os.path.splitext(video_name)[0]
    video_out = os.path.join(OUTPUT_DIR, video_id)
    I_dir = os.path.join(video_out, "I")
    P_dir = os.path.join(video_out, "P")
    B_dir = os.path.join(video_out, "B")
    os.makedirs(I_dir, exist_ok=True)
    os.makedirs(P_dir, exist_ok=True)
    os.makedirs(B_dir, exist_ok=True)
    frame_types = get_frame_types(video_path)
    tmp_video_dir = os.path.join(TMP_DIR, video_id)
    extract_all_frames(video_path, tmp_video_dir)
    frames = sorted(os.listdir(tmp_video_dir))

    for idx, frame_name in enumerate(tqdm(frames, desc="Sorting frames")):
        if idx >= len(frame_types):
            break

        frame_type = frame_types[idx].strip()
        src = os.path.join(tmp_video_dir, frame_name)

        if frame_type == "I":
            dst = os.path.join(I_dir, frame_name)
        elif frame_type == "P":
            dst = os.path.join(P_dir, frame_name)
        elif frame_type == "B":
            dst = os.path.join(B_dir, frame_name)
        else:
            continue

        shutil.move(src, dst)

    shutil.rmtree(tmp_video_dir)

print("\n✅ All videos processed successfully")
