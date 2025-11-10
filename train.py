from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from vatex_dataset import VATEXReadyDataset
from hgat_model import SimpleHGAT
from load_annotations import annotations_list  # Load your annotations list here

def collate_fn(batch):
    feats, edges, input_ids, att_mask = zip(*batch)
    feats_padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    input_ids = torch.stack(input_ids)
    att_mask = torch.stack(att_mask)
    return feats_padded, edges, input_ids, att_mask

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = VATEXReadyDataset("features", annotations_list, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_feats, _, _, _ = dataset[0]
feat_dim = sample_feats.shape[1]
print(f"Detected feature dimension: {feat_dim}")

model = SimpleHGAT(feat_dim=feat_dim, hidden_dim=512, vocab_size=tokenizer.vocab_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

for epoch in range(50):
    total_correct = 0
    total_tokens = 0
    for feats, edges, input_ids, att_mask in loader:
        feats, input_ids = feats.to(device), input_ids.to(device)
        preds = model(feats)
        min_len = min(preds.size(1), input_ids.size(1))

        preds_slice = preds[:, :min_len, :]
        input_ids_slice = input_ids[:, :min_len]

        loss = F.cross_entropy(
            preds_slice.contiguous().view(-1, preds.size(-1)),
            input_ids_slice.contiguous().view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy of predictions ignoring batch dimension by comparing top predictions with ground truth
        pred_tokens = preds_slice.argmax(dim=-1)
        correct = (pred_tokens == input_ids_slice).sum().item()
        total_correct += correct
        total_tokens += input_ids_slice.numel()

    epoch_acc = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"Epoch {epoch} Loss: {loss.item()} Accuracy: {epoch_acc:.4f}")
