import os
import cv2
import numpy as np
import joblib
from insightface.model_zoo import get_model

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loading arcface model
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully")


# generating embeddings
def generate_embeddings(dataset_path):
    X = []
    y = []
    for person in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder):
            continue

        # path for pre-processing the image
        print(f"Processing {person}")
        for image_file in sorted(os.listdir(person_folder)):
            image_path = os.path.join(person_folder, image_file)
            img = cv2.imread(image_path)
            if img is None:
                continue

            # pre-processing the image
            # Light denoising
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize for ArcFace
            img = cv2.resize(img, (112, 112))

            # arcface embeddings
            embedding = arcface.get_feat(img)[0]
            X.append(embedding)
            y.append(person)
    return np.array(X), np.array(y)


# generating training embeddings
print("\nGenerating Training Embeddings...\n")
X_train, y_train = generate_embeddings("train")
print(f"\nTraining Embeddings Generated: {len(X_train)}")
print("Training embedding shape:", X_train.shape)

# generating testing embeddings
print("\nGenerating Testing Embeddings...\n")
X_test, y_test = generate_embeddings("test")
print(f"\nTesting Embeddings Generated: {len(X_test)}")
print("Testing embedding shape:", X_test.shape)

# svm training
print("\nTraining SVM...\n")
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)
print("Training Complete")

# testing svm model
predictions = svm.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))
print("\nFace Recognition Completed Successfully.")
