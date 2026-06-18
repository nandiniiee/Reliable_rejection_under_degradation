#this program is used for comparing two faces (1:1 comparing)

import os
import cv2
from insightface.model_zoo import get_model
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace


# extracting face and cropping
def extract_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    faces = RetinaFace.detect_faces(img)
    if not isinstance(faces, dict):
        raise ValueError(f"No face detected in {image_path}")

    # take best detected face
    first_face = list(faces.values())[0]
    x1, y1, x2, y2 = first_face["facial_area"]
    face = img[y1:y2, x1:x2]
    return face


# loading arcface model
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully")


# getting embeddings
def get_embedding(image_path):
    img = extract_face(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # preprocessing
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    embedding = arcface.get_feat(img)[0]
    return embedding


# image paths
image1_path = "probe_images/gates.jpg"
image2_path = "database_images/billGates.jpg"

# embeddings for imput images
embedding1 = get_embedding(image1_path)
embedding2 = get_embedding(image2_path)

# cosine similarity
similarity = cosine_similarity([embedding1], [embedding2])[0][0]
print("\nCosine Similarity:", round(similarity, 4))

# threshold
threshold = 0.75
if similarity >= threshold:
    print("\nSame Person")
else:
    print("\nDifferent Persons")
