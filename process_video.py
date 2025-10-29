import cv2
from ultralytics import YOLO
import yt_dlp
import os
import argparse
import csv
import numpy as np

# YOLOv8 keypoint mapping
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]
# Adjust keypoint indices to be 0-based
SKELETON = [[p[0] - 1, p[1] - 1] for p in SKELETON]

# Colors for keypoints and skeleton
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (255, 0, 0)  # Blue

def main(video_url):
    # --- 1. Download Video ---
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'downloaded_video.%(ext)s',
        'ffmpeg_location': './ffmpeg'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_path = ydl.prepare_filename(info)

    # --- 2. Process Video ---
    model = YOLO('yolov8l-pose.pt')
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation.mp4', fourcc, fps, (width, height))

    with open('keypoints.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_id', 'person_id', 'keypoint_id', 'x', 'y', 'confidence'])

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            # Create a black background
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            keypoints_data = results[0].keypoints.cpu().numpy()

            for person_id, person_keypoints in enumerate(keypoints_data.data):
                # Draw skeleton
                for p1_idx, p2_idx in SKELETON:
                    if p1_idx < len(person_keypoints) and p2_idx < len(person_keypoints):
                        x1, y1, conf1 = person_keypoints[p1_idx]
                        x2, y2, conf2 = person_keypoints[p2_idx]
                        if conf1 > 0.5 and conf2 > 0.5: # Draw only if confident
                            cv2.line(black_frame, (int(x1), int(y1)), (int(x2), int(y2)), SKELETON_COLOR, 2)

                # Draw keypoints
                for keypoint_id, (x, y, conf) in enumerate(person_keypoints):
                    if conf > 0.5: # Draw only if confident
                        cv2.circle(black_frame, (int(x), int(y)), 5, KEYPOINT_COLOR, -1)

                    # Write to CSV
                    csv_writer.writerow([frame_id, person_id, keypoint_id, x, y, conf])

            out.write(black_frame)
            frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a YouTube video to detect player movements.")
    parser.add_argument("url", help="The URL of the YouTube video to process.")
    args = parser.parse_args()

    main(args.url)
