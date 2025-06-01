# check_labels.py
import json

print("Checking imagenet_classes.json format...")
with open('imagenet_classes.json', 'r') as f:
    labels = json.load(f)
    
print(f"Type of labels: {type(labels)}")
print(f"First few items: {list(labels.items())[:5] if isinstance(labels, dict) else labels[:5]}")