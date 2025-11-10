import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Use modern weights argument instead of deprecated pretrained=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    return features.cpu().numpy()

frames_dir = "frames"
feature_save_dir = "features"
os.makedirs(feature_save_dir, exist_ok=True)

for video_id in os.listdir(frames_dir):
    video_frame_dir = os.path.join(frames_dir, video_id)
    feature_output_path = os.path.join(feature_save_dir, f"{video_id}.npy")
    # SKIP if already processed
    if os.path.isfile(feature_output_path):
        print(f"Features already extracted for {video_id}, skipping.")
        continue
    video_features = []
    for frame_file in sorted(os.listdir(video_frame_dir)):
        frame_path = os.path.join(video_frame_dir, frame_file)
        feat = extract_features(frame_path)
        video_features.append(feat)
    video_features = np.vstack(video_features)
    np.save(feature_output_path, video_features)
    print(f"Saved features for {video_id}")
