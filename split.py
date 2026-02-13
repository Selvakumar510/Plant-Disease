import os
import shutil
import hashlib
import random

# ------------------------
# CONFIG
# ------------------------
DATASET_DIR = "C:\\Users\\jothe\\OneDrive\\Documents\\PlantDisease\\backend\\data\\full"       # Original dataset
OUTPUT_DIR = "data_split"     # Split output
RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}  # Split ratios

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def hash_file(file_path):
    """Return md5 hash of file"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ------------------------
# MAIN LOGIC
# ------------------------
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {class_name}")
    
    # 1️⃣ Group duplicates by hash
    hash_groups = {}
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        if not os.path.isfile(img_path):
            continue
        img_hash = hash_file(img_path)
        if img_hash in hash_groups:
            hash_groups[img_hash].append(img_path)
        else:
            hash_groups[img_hash] = [img_path]
    
    # Each hash group is treated as a single leaf/group
    groups = list(hash_groups.values())
    random.shuffle(groups)

    # 2️⃣ Split groups into train/val/test
    total = len(groups)
    train_end = int(total * RATIO["train"])
    val_end = train_end + int(total * RATIO["val"])
    
    splits = {
        "train": groups[:train_end],
        "val": groups[train_end:val_end],
        "test": groups[val_end:]
    }

    # 3️⃣ Copy images to output folders
    for split_name, split_groups in splits.items():
        for group in split_groups:
            for img_path in group:
                dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
                ensure_dir(dest_dir)
                shutil.copy(img_path, dest_dir)
    
    # Print summary
    print(f"Total groups: {total}")
    for split_name, split_groups in splits.items():
        count = sum(len(g) for g in split_groups)
        print(f"{split_name}: {count} images in {len(split_groups)} group(s)")
