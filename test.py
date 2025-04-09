import os

# Use full path to the dataset
dataset_path = "C:/Users/KIIT0001/Desktop/Brain-Tumor-Detection/dataset/training"

categories = ["glioma", "meningioma", "pituitary", "no_tumor"]

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        print(f"❌ Folder missing: {folder_path}")
    else:
        images = os.listdir(folder_path)
        print(f"✅ {category}: Found {len(images)} images")
