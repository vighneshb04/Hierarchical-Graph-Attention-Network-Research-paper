from datasets import Dataset
import json

# Load your VATEX annotation Arrow file (change path to your actual file)
ds = Dataset.from_file("json/validation/data-00000-of-00001.arrow")

annotations_list = []

# Loop through dataset and extract videoID + enCap captions
for record in ds:
    annotations_list.append({
        "videoID": record["videoID"],
        "enCap": record["enCap"],
        "chCap": record["chCap"],
        "start": record.get("start", None),
        "end": record.get("end", None),
        "path": record.get("path", None)
    })

# Save to local JSON for easy next step
with open("annotations_extracted.json", "w", encoding="utf-8") as f:
    json.dump(annotations_list, f, ensure_ascii=False)

print(f"Saved {len(annotations_list)} annotation entries to annotations_extracted.json")
