# FACE RECOGNITION PROJECT
# Classifier Based Approach using ArcFace + SVM
# Experiments 1 & 2: Closed Set Accuracy + Calibration

import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import insightface
from insightface.app import FaceAnalysis
from retinaface import RetinaFace

print("All imports done!")

# configuring paths
CROPPED_FACES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cropped_faces"
)

PERSONS = ["gates", "jack", "modi", "musk", "trump"]
EMBEDDINGS_PATH = "embeddings.npy"
LABELS_PATH = "labels.npy"
SVM_MODEL_PATH = "svm_model.pkl"
LABEL_ENC_PATH = "label_encoder.pkl"

# loading arcface model
print("\nLoading ArcFace...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
rec_model = app.models["recognition"]
print("ArcFace loaded!")

# extracting the embeddings or loading the saved embeddings
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(LABELS_PATH):
    print("\nFound saved embeddings — loading...")
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    print(f"Loaded {len(embeddings)} embeddings!")

else:
    print("\nExtracting embeddings from cropped faces...")
    embeddings = []
    labels = []

    for person in PERSONS:
        person_dir = os.path.join(CROPPED_FACES_DIR, person)
        if not os.path.exists(person_dir):
            print(f"  WARNING: folder not found for {person}")
            continue

        photos = os.listdir(person_dir)
        print(f"  Processing {person} ({len(photos)} photos)...")

        for photo in photos:
            img_path = os.path.join(person_dir, photo)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize to 112x112 ArcFace required input size
            img_resized = cv2.resize(img, (112, 112))
            # Convert BGR to RGB — ArcFace expects RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Get 512-dim embedding
            embedding = rec_model.get_feat(img_rgb).flatten()
            embeddings.append(embedding)
            labels.append(person)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Save for next time
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)
    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Shape: {embeddings.shape}")
    print("Saved embeddings.npy and labels.npy!")


# train svm and save it or load the saved model
if os.path.exists(SVM_MODEL_PATH) and os.path.exists(LABEL_ENC_PATH):
    print("\nFound saved model — loading...")
    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)
    with open(LABEL_ENC_PATH, "rb") as f:
        le = pickle.load(f)
    y = le.transform(labels)
    _, X_test, _, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Model loaded!")

else:
    print("\nTraining SVM...")
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred) * 100

    print(f"\nBaseline accuracy (clean): {round(baseline_acc, 2)}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(svm, f)
    with open(LABEL_ENC_PATH, "wb") as f:
        pickle.dump(le, f)

    print("Model saved!")



# Degradation
# Blur - Gaussian blur (kernel 3, 7, 15)
# Compression - JPEG quality (90%, 50%, 10%)
# Resolution - Downscale then upscale (0.5x, 0.25x, 0.1x)
def apply_blur(img, level):
    kernel_sizes = {1: 3, 2: 7, 3: 15}
    k = kernel_sizes[level]
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_compression(img, level):
    qualities = {1: 90, 2: 50, 3: 10}
    q = qualities[level]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, 1)

def apply_resolution(img, level):
    scales = {1: 0.5, 2: 0.25, 3: 0.1}
    s = scales[level]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    return cv2.resize(small, (w, h))

print("\nDegradation functions ready!")

# clean photo accuracy test
print("\n" + "=" * 50)
print("CLEAN PHOTO TEST")
print("=" * 50)
test_path = input("Enter path to test image: ").strip()

if not os.path.exists(test_path):
    print(f"File not found: {test_path}")
else:
    test_img = cv2.imread(test_path)
    faces = RetinaFace.detect_faces(test_path)

    if isinstance(faces, dict) and len(faces) > 0:
        x1, y1, x2, y2 = faces[list(faces.keys())[0]]["facial_area"]
        crop = test_img[y1:y2, x1:x2]

        img_resized = cv2.resize(crop, (112, 112))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        embedding = rec_model.get_feat(img_rgb).flatten().reshape(1, -1)

        prediction = svm.predict(embedding)
        probability = svm.predict_proba(embedding)
        name = le.inverse_transform(prediction)[0]
        confidence = round(max(probability[0]) * 100, 2)

        print(f"\nThis is: {name.upper()} (Confidence: {confidence}%)")

        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected: {name.upper()} ({confidence}%)")
        plt.axis("off")
        plt.show()

    else:
        print("No face detected! Try a clearer photo.")

# Experiments 1 and 2
# Experiment 1: Closed Set Accuracy Under Degradation
#   → Image grid showing predictions at each degradation level
# Experiment 2: Calibration Under Degradation
#   → Line graphs of confidence vs degradation level
print("\n" + "=" * 50)
print("EXPERIMENTS 1 & 2 — DEGRADATION TESTING")
print("=" * 50)
exp_path = input("Enter path to image for experiments: ").strip()

if not os.path.exists(exp_path):
    print(f"File not found: {exp_path}")
else:
    test_img = cv2.imread(exp_path)
    faces = RetinaFace.detect_faces(exp_path)

    if isinstance(faces, dict) and len(faces) > 0:
        x1, y1, x2, y2 = faces[list(faces.keys())[0]]["facial_area"]
        clean_crop = test_img[y1:y2, x1:x2]

        # Clean baseline
        img_resized = cv2.resize(clean_crop, (112, 112))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        emb = rec_model.get_feat(img_rgb).flatten().reshape(1, -1)
        pred = svm.predict(emb)
        prob = svm.predict_proba(emb)
        true_name = le.inverse_transform(pred)[0]
        clean_conf = round(max(prob[0]) * 100, 2)

        print(f"\nPerson: {true_name.upper()} | " f"Clean Confidence: {clean_conf}%")

        all_confs = {"Clean": clean_conf}
        all_preds = {"Clean": true_name}
        all_crops = {"Clean": clean_crop}

        degradation_types = {
            "Blur": apply_blur,
            "Compression": apply_compression,
            "Resolution": apply_resolution,
        }
        level_names = {1: "Mild", 2: "Medium", 3: "Severe"}

        # Apply all degradations
        for deg_name, deg_func in degradation_types.items():
            for level in [1, 2, 3]:
                # Degrade FULL image 
                img_deg = deg_func(test_img, level)
                temp_path = "temp_deg.jpg"
                cv2.imwrite(temp_path, img_deg)

                # RetinaFace on degraded image
                faces_deg = RetinaFace.detect_faces(temp_path)

                if isinstance(faces_deg, dict) and len(faces_deg) > 0:
                    x1d, y1d, x2d, y2d = faces_deg[list(faces_deg.keys())[0]][
                        "facial_area"
                    ]
                    crop_deg = img_deg[y1d:y2d, x1d:x2d]
                else:
                    # Fallback to original crop coordinates
                    crop_deg = img_deg[y1:y2, x1:x2]
                    print(f"  RetinaFace failed: " f"{deg_name} level {level}")

                # ArcFace + SVM
                img_resized = cv2.resize(crop_deg, (112, 112))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                emb = rec_model.get_feat(img_rgb).flatten().reshape(1, -1)
                pred = svm.predict(emb)
                prob = svm.predict_proba(emb)
                pred_name = le.inverse_transform(pred)[0]
                conf = round(max(prob[0]) * 100, 2)

                key = f"{deg_name} {level_names[level]}"
                all_confs[key] = conf
                all_preds[key] = pred_name
                all_crops[key] = crop_deg

        # Experiment 1 image grid
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for idx, (key, crop) in enumerate(all_crops.items()):
            pred_name = all_preds[key]
            conf = all_confs[key]
            correct = "✓" if pred_name == true_name else "✗"
            axes[idx].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"{key}\n{pred_name.upper()} {conf}% {correct}")
            axes[idx].axis("off")

        plt.suptitle(
            f"Experiment 1 — Closed Set Accuracy: {true_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # Experiment 2: Calibration line graphs
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax_idx, (deg_name, color) in enumerate(
            zip(["Blur", "Compression", "Resolution"], ["blue", "green", "red"])
        ):

            confs = [
                clean_conf,
                all_confs[f"{deg_name} Mild"],
                all_confs[f"{deg_name} Medium"],
                all_confs[f"{deg_name} Severe"],
            ]
            preds = [
                true_name,
                all_preds[f"{deg_name} Mild"],
                all_preds[f"{deg_name} Medium"],
                all_preds[f"{deg_name} Severe"],
            ]

            axes[ax_idx].plot(
                ["Clean", "Mild", "Medium", "Severe"],
                confs,
                f"{color[0]}-o",
                linewidth=2,
                markersize=8,
            )
            axes[ax_idx].axhline(
                y=50, color="red", linestyle="--", label="50% threshold"
            )

            for i, (conf, pred) in enumerate(zip(confs, preds)):
                correct = "✓" if pred == true_name else "✗"
                axes[ax_idx].annotate(
                    f"{conf}%\n{pred.upper()}{correct}",
                    (i, conf),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                )

            axes[ax_idx].set_title(f"{deg_name} Degradation")
            axes[ax_idx].set_ylabel("Confidence (%)")
            axes[ax_idx].set_ylim(0, 120)
            axes[ax_idx].legend()

        plt.suptitle(
            f"Experiment 2 — Calibration: " f"{true_name.upper()} under Degradation",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        #summary table
        print(
            f"\n{'Type':<25} {'Confidence':>12} " f"{'Predicted':>12} {'Correct?':>10}"
        )
        print("-" * 62)
        for k, v in all_confs.items():
            pred = all_preds[k]
            correct = "✅" if pred == true_name else "❌"
            print(f"{k:<25} {v:>11}% " f"{pred.upper():>12} {correct:>10}")

    else:
        print("No face detected in image!")

# Clean up temp file
if os.path.exists("temp_deg.jpg"):
    os.remove("temp_deg.jpg")

print("\nDone!")
