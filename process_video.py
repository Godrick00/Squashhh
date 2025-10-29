import cv2
from ultralytics import YOLO
import yt_dlp
import os
import argparse
import csv

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Changed codec for .mp4
    out = cv2.VideoWriter('animation.mp4', fourcc, fps, (width, height))

    # Open CSV file for writing keypoints
    with open('keypoints.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['frame_id', 'person_id', 'keypoint_id', 'x', 'y', 'confidence'])

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run pose estimation
            results = model(frame)

            # Draw annotations
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # Extract and write keypoints to CSV
            keypoints = results[0].keypoints.cpu().numpy()
            for person_id, person_keypoints in enumerate(keypoints.data):
                for keypoint_id, (x, y, conf) in enumerate(person_keypoints):
                    csv_writer.writerow([frame_id, person_id, keypoint_id, x, y, conf])

            frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- 3. Cleanup ---
    if os.path.exists(video_path):
        os.remove(video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a YouTube video to detect player movements.")
    parser.add_argument("url", help="The URL of the YouTube video to process.")
    args = parser.parse_args()

    main(args.url)
