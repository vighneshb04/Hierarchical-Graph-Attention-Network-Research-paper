import torch
from transformers import BertTokenizer
import numpy as np
import os

from hgat_model import SimpleHGAT  # Import your model class

def test_single_video(video_id, features_dir="features"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    features_path = os.path.abspath(os.path.join(features_dir, f"{video_id}.npy"))

    if not os.path.isfile(features_path):
        print(f"Feature file not found: {features_path}")
        return

    print(f"Loading: {features_path}")

    features = np.load(features_path)
    features = torch.from_numpy(features).float().unsqueeze(0).to(device)  # (1, T, F)

    feat_dim = features.shape[2]
    print(f"Detected feature dimension: {feat_dim}")

    model = SimpleHGAT(feat_dim=feat_dim, hidden_dim=512, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(features)
        pred_tokens = preds.argmax(dim=-1).squeeze(0).cpu().numpy()

    print(f"Input feature shape: {features.shape}")
    print("Predicted token indices:", pred_tokens)
    tokens_str = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    print(f"Predicted caption text: {tokens_str}")

if __name__ == "__main__":
    # Put a valid video id here (such as '_kAIMDkIKdw')
    test_single_video("_9JpiZ5gHvA", features_dir="features")
