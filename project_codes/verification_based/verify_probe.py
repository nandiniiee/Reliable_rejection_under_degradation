import os
import cv2
import pickle
from retinaface import RetinaFace
from insightface.model_zoo import get_model
from sklearn.metrics.pairwise import cosine_similarity

# Load ArcFace
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully")

# Load Gallery Embeddings
with open("gallery_embeddings.pkl", "rb") as f:
    gallery_embeddings = pickle.load(f)
print("Gallery embeddings loaded successfully")

# Probe Image Path
probe_image_path = "probe_images/probe1.jpg"
img = cv2.imread(probe_image_path)
if img is None:
    print("Probe image not found")
    exit()

# Face Detection using RetinaFace
faces = RetinaFace.detect_faces(img)
if not isinstance(faces, dict) or len(faces) == 0:
    print("No face detected in probe image")
    exit()

# Select face with highest confidence
best_face = max(faces.values(), key=lambda face: face["score"])
confidence = best_face["score"]
print(f"Detection Confidence: {confidence:.4f}")

# Crop Face
x1, y1, x2, y2 = best_face["facial_area"]
h, w = img.shape[:2]
x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(w, x2)
y2 = min(h, y2)
face_crop = img[y1:y2, x1:x2]
if face_crop.size == 0:
    print("Invalid face crop")
    exit()

# Preprocessing for ArcFace
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
face_crop = cv2.resize(face_crop, (112, 112))

# Generate Probe Embedding
probe_embedding = arcface.get_feat(face_crop)[0]

# Compare Against Gallery embeddings
best_person = None
best_similarity = -1
print("\nSimilarity Scores")

for person, gallery_embedding in gallery_embeddings.items():
    similarity = cosine_similarity([probe_embedding], [gallery_embedding])[0][0]
    print(f"{person:<15} : {similarity:.4f}")
    if similarity > best_similarity:
        best_similarity = similarity
        best_person = person

# Final Decision
threshold = 0.70
print(f"\nBest Match      : {best_person}")
print(f"\nSimilarity Score: {best_similarity:.4f}")
if best_similarity >= threshold:
    print("\nVERIFIED")
    print(f"Identity : {best_person}")
else:
    print("\nUNKNOWN PERSON")
