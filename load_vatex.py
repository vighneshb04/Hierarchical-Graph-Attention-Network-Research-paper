from datasets import load_dataset

# Replace with the actual path to your json/train folder
dataset = load_dataset('json', data_files={"train": "json/train/data-00000-of-00001.arrow"})

print(dataset['train'])
print("First example:", dataset['train'][0])
