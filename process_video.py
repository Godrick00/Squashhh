import cv2
from ultralytics import YOLO
import yt_dlp
import os
import argparse

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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

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
