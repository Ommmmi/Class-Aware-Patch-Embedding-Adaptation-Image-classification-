import os
import shutil
import pandas as pd

# Set paths
image_dir = r'C:\Users\Administrator\Downloads\CPEA-main\Kvasir dataset\images_by_label'
output_base = r'C:\Users\Administrator\Downloads\CPEA-main\split_data_for_kvasir'
os.makedirs(output_base, exist_ok=True)

# Load CSVs
csv_folder = r'C:\Users\Administrator\Downloads\CPEA-main\Kvasir dataset'
train_df = pd.read_csv(os.path.join(csv_folder, 'train.csv'))
val_df = pd.read_csv(os.path.join(csv_folder, 'val.csv'))
test_df = pd.read_csv(os.path.join(csv_folder, 'test.csv'))


# Build lookup of image name -> full path
image_paths = {}
for root, _, files in os.walk(image_dir):
    for fname in files:
        if fname.endswith('.jpg'):
            image_paths[fname] = os.path.join(root, fname)

# Copying helper
def copy_images(df, split_name):
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)

    found, missing, errors = 0, 0, 0
    for image_name in df['image_id']:
        src_path = image_paths.get(image_name)
        dst_path = os.path.join(split_dir, image_name)

        if not src_path:
            print(f"❌ Missing: {image_name}")
            missing += 1
            continue

        try:
            if os.path.isdir(dst_path):
                print(f"⚠️ Skipping directory named like file: {dst_path}")
                errors += 1
                continue
            shutil.copy2(src_path, dst_path)
            found += 1
        except Exception as e:
            print(f"❌ Error copying {image_name}: {e}")
            errors += 1

    print(f"\n📂 {split_name.upper()} SUMMARY:")
    print(f"   ✅ Copied: {found}")
    print(f"   ❌ Missing: {missing}")
    print(f"   ⚠️ Errors: {errors}\n")

# Perform splits
copy_images(train_df, 'train')
copy_images(val_df, 'val')
copy_images(test_df, 'test')

print("✅ Done: All splits processed.")
