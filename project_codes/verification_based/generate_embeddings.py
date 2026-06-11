#program used for generating embeddings of our cropped faces using arcface
# and saving them in a dictionary for future use

import os
import cv2
import pickle
import numpy as np
from insightface.model_zoo import get_model

# Load ArcFace
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)

print("ArcFace loaded successfully")
gallery_embeddings = {}
gallery_path = "cropped_faces"
for image_file in os.listdir(gallery_path):
    image_path = os.path.join(gallery_path, image_file)
    img = cv2.imread(image_path)
    if img is None:
        continue

    #pre-processing
    name = os.path.splitext(image_file)[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))

    #generating embedding
    embedding = arcface.get_feat(img)[0]
    gallery_embeddings[name]=embedding
    print("\nDictionary keys:")
    print(gallery_embeddings.keys())
    print(f"Generated embedding for {name}")
print(len(gallery_embeddings))

# Save embeddings
with open("gallery_embeddings.pkl", "wb") as f:
    pickle.dump(gallery_embeddings, f)

# Read it back
with open("gallery_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print("\nGallery embeddings saved successfully.")
print(f"Total identities: {len(gallery_embeddings)}")
