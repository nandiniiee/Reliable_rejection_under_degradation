import os
import cv2
import random
import numpy as np
from insightface.model_zoo import get_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# laoding arcface model
model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully")


# generating embeddings
def generate_embeddings(dataset_path):
    embeddings = {}
    for person in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder):
            continue
        embeddings[person] = []
        print(f"Processing {person}")
        for image_file in sorted(os.listdir(person_folder)):
            image_path = os.path.join(person_folder, image_file)
            img = cv2.imread(image_path)
            if img is None:
                continue

            # image pre-processing
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))

            # arcface embeddings
            embedding = arcface.get_feat(img)
            if len(embedding.shape) > 1:
                embedding = embedding[0]
            embeddings[person].append(embedding)
    return embeddings


# creating positive and negative pairs
def create_pairs(embeddings):
    similarities = []
    labels = []
    persons = list(embeddings.keys())

    # positive pairs
    for person in persons:
        person_embeddings = embeddings[person]
        for i in range(len(person_embeddings) - 1):
            emb1 = person_embeddings[i]
            emb2 = person_embeddings[i + 1]
            sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(sim)
            labels.append(1)

    # negative paris
    num_positive = len(labels)
    for _ in range(num_positive):
        p1, p2 = random.sample(persons, 2)
        emb1 = random.choice(embeddings[p1])
        emb2 = random.choice(embeddings[p2])
        sim = cosine_similarity([emb1], [emb2])[0][0]
        similarities.append(sim)
        labels.append(0)
    return similarities, labels


# training embeddings to find cosine threshold
print("\nGenerating Train Embeddings...\n")
train_embeddings = generate_embeddings("train")

# training pairs
train_similarities, train_labels = create_pairs(train_embeddings)

# finding best threshold
best_threshold = 0
best_accuracy = 0
for threshold in np.arange(0.30, 0.95, 0.01):
    predictions = []
    for sim in train_similarities:
        if sim >= threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    accuracy = accuracy_score(train_labels, predictions)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print("\nBest Threshold:", round(best_threshold, 2))
print("Training Verification Accuracy:", round(best_accuracy * 100, 2), "%")


# testing embeddings with new threshold
print("\nGenerating Test Embeddings...\n")
test_embeddings = generate_embeddings("test")
test_similarities, test_labels = create_pairs(test_embeddings)

# testing verification
predictions = []
for sim in test_similarities:
    if sim >= best_threshold:
        predictions.append(1)
    else:
        predictions.append(0)

accuracy = accuracy_score(test_labels, predictions)
print("\nVerification Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(test_labels, predictions))
print("\nConfusion Matrix:\n")
print(confusion_matrix(test_labels, predictions))
print("\nFace Verification Completed Successfully.")
