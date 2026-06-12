#old program we wrote where we generate the database embeddings everytime this program runs
#not saving the databse embeddings
#also verifying with the probe image and giving a similarity score

import os
import cv2
import numpy as np
from insightface.model_zoo import get_model
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# loading arcface model
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully")


# generating database embeddings
def generate_embeddings(dataset_path):
    embeddings = {}
    for person in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder):
            continue
        embeddings[person] = []
        print(f"Processing {person}")
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            img = cv2.imread(image_path)
            if img is None:
                continue

            # preprocessing
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            embedding = arcface.get_feat(img)[0]
            embeddings[person].append(embedding)
    return embeddings


# database
print("\nGenerating database embeddings...\n")
database_embeddings = generate_embeddings("train")

# query image path
query_image_path = "query_images/yuvaansh.jpeg"
img = cv2.imread(query_image_path)
if img is None:
    print("Query image not found")
    exit()
img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))
query_embedding = arcface.get_feat(img)[0]

# find best match
best_similarity = -1
best_person = None
for person in database_embeddings:
    for emb in database_embeddings[person]:
        similarity = cosine_similarity([query_embedding], [emb])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_person = person

# threshold
threshold = 0.75
print("\nBest Match :", best_person)
print("Similarity :", round(best_similarity, 4))
if best_similarity >= threshold:
    print("\nRecognized Person:", best_person)
else:
    print("\nUnknown Person")
