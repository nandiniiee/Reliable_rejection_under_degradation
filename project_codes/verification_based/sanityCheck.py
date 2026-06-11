import os
import cv2
import random
import numpy as np
from insightface.model_zoo import get_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from retinaface import RetinaFace
from insightface.model_zoo import get_model
# Load ArcFace
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully")

#sanity check
img = cv2.imread("cropped_faces/billGates.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))
emb1 = arcface.get_feat(img)[0]
emb2 = arcface.get_feat(img)[0]
similarity = cosine_similarity([emb1], [emb2])[0][0]

print(similarity)
