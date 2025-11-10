import json

def load_vatex_annotations():
    with open("annotations_extracted.json", "r", encoding="utf-8") as f:
        annotations_list = json.load(f)
    return annotations_list

annotations_list = load_vatex_annotations()
