import os
import cv2

videos_dir = 'videos'
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]

for file in video_files:
    video_id = os.path.splitext(file)[0]
    video_path = os.path.join(videos_dir, file)
    out_dir = os.path.join(frames_dir, video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    second = 0
    while second < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{out_dir}/frame_{int(second):04d}.jpg", frame)
        second += 1
    cap.release()
    print(f"Extracted frames for {video_id}")
