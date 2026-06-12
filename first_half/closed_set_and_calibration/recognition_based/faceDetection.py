import cv2
import os
import time  # to measure processing time

# using retinaFace model here
# retinaFace gives 5 facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner) which can be used for better alignment and cropping of faces.
from retinaface import RetinaFace
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices("GPU"))

# input and output folder
input_root = "dataset_fiveFaces"
output_root = "cropped_faces"

os.makedirs(output_root, exist_ok=True)

# measuring time
total_start = time.time()

for root, dirs, files in os.walk(input_root):

    # Preserving directory structure
    relative_path = os.path.relpath(root, input_root)
    output_dir = os.path.join(output_root, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    for file in files:

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        input_path = os.path.join(root, file)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read image: {input_path}")
            continue
        try:
            faces = RetinaFace.detect_faces(img)

            # for case where no face detected
            if not isinstance(faces, dict) or len(faces) == 0:
                print(f"No face detected in image: {input_path}")
                continue

            # selecting face with highest confidence score
            best_face = max(faces.values(), key=lambda face: face["score"])
            x1, y1, x2, y2 = best_face["facial_area"]
            h, w = img.shape[:2]

            # padding
            face_size = max(x2 - x1, y2 - y1)
            pad = int(0.05 * face_size)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # cropped face
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"Invalid crop for image: {input_path}")
                continue

            # saving the cropped face
            save_path = os.path.join(output_dir, file)
            cv2.imwrite(save_path, crop)
            print(f"Cropped face saved to: {save_path}")

        except Exception as e:
            print(f"Error processing image {input_path}: {e}")
            print(e)

total_end = time.time()
print(f"Total processing time: {total_end-total_start:.2f} seconds")
print("Face detection and cropping completed.")
