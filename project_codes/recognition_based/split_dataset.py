import os
import shutil
from sklearn.model_selection import train_test_split

# Original dataset
dataset_path = "cropped_faces"

# Output folders
train_root = "train"
test_root = "test"
os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue
    images = []
    for file in os.listdir(person_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(file)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    train_person_dir = os.path.join(train_root, person)
    test_person_dir = os.path.join(test_root, person)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(person_folder, img)
        dst = os.path.join(train_person_dir, img)
        shutil.copy2(src, dst)

    # Copy testing images
    for img in test_images:
        src = os.path.join(person_folder, img)
        dst = os.path.join(test_person_dir, img)
        shutil.copy2(src, dst)

    print(f"{person}: " f"{len(train_images)} train, " f"{len(test_images)} test")

print("\nDataset split completed.")
