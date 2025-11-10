import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VATEXReadyDataset(Dataset):
    def __init__(self, features_dir, annotations_list, tokenizer):
        self.annotations = [entry for entry in annotations_list
                           if os.path.isfile(os.path.join(features_dir, f"{entry['videoID']}.npy"))]
        self.features_dir = features_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        entry = self.annotations[idx]
        video_id = entry['videoID']
        features = np.load(os.path.join(self.features_dir, f"{video_id}.npy"))
        caption = entry['enCap'][0]
        cap_tokens = self.tokenizer(caption, return_tensors='pt', padding='max_length', max_length=20, truncation=True)
        edges = [(i, i + 1) for i in range(features.shape[0] - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return torch.from_numpy(features).float(), edge_index, cap_tokens['input_ids'].squeeze(), cap_tokens['attention_mask'].squeeze()
