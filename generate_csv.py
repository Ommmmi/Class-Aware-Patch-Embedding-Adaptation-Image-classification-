import os
import pandas as pd

# Path to your Kvasir dataset
DATASET_PATH = "C:\\Users\\Administrator\\Downloads\\CPEA-main\\Kvasir dataset\\images_by_label"

# Class splits
train_classes = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis']
test_classes = ['polyps', 'ulcerative-colitis']
val_classes = ['normal-cecum', 'normal-pylorus', 'normal-z-line']

def generate_csv(class_list, split_name):
    records = []
    for cls in class_list:
        class_dir = os.path.join(DATASET_PATH, cls)
        if not os.path.exists(class_dir):
            print(f" Warning: Directory {class_dir} not found!")
            continue
        for img in os.listdir(class_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append([img, cls])
    df = pd.DataFrame(records, columns=["image_id", "dx"])
    df.to_csv(f"{split_name}.csv", index=False)
    print(f" {split_name}.csv saved with {len(df)} samples.")

# Generate CSVs
generate_csv(train_classes, "train")
generate_csv(test_classes, "test")
generate_csv(val_classes, "val")
