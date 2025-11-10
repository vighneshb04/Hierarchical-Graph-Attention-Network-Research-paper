import os
from extract_feature import extract_features  # your existing extract function
import numpy as np

frames_dir = "frames"
feature_save_dir = "features"
os.makedirs(feature_save_dir, exist_ok=True)

for video_id in os.listdir(frames_dir):
    feature_output_path = os.path.join(feature_save_dir, f"{video_id}.npy")
    if os.path.isfile(feature_output_path):
        continue  # Skip already extracted

    video_frame_dir = os.path.join(frames_dir, video_id)
    video_features = []
    for frame_file in sorted(os.listdir(video_frame_dir)):
        frame_path = os.path.join(video_frame_dir, frame_file)
        feat = extract_features(frame_path)
        video_features.append(feat)
    video_features = np.vstack(video_features)
    np.save(feature_output_path, video_features)
    print(f"Saved features for {video_id}")
