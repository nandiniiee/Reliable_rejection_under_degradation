# FACE VERIFICATION PROJECT
# Verification Based Approach using ArcFace + Cosine Similarity
# Experiments 1 & 2: Closed Set Accuracy + Calibration
# Run: python verify_probe.py

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from insightface.model_zoo import get_model
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1 — LOAD ARCFACE MODEL
model_path = os.path.expanduser(
    "~/.insightface/models/buffalo_l/w600k_r50.onnx")
arcface = get_model(model_path)
arcface.prepare(ctx_id=-1)
print("ArcFace loaded successfully!")

# ============================================================
# STEP 2 — LOAD GALLERY EMBEDDINGS
# ============================================================
with open("gallery_embeddings.pkl", "rb") as f:
    gallery_embeddings = pickle.load(f)
print(f"Gallery loaded! People: {list(gallery_embeddings.keys())}")

# ============================================================
# STEP 3 — CONFIGURATION
# ============================================================
THRESHOLD   = 0.70   # below this = unknown person
PROBE_PATH  = input("Enter path to probe image: ").strip()

# ============================================================
# STEP 4 — HELPER FUNCTIONS
# ============================================================

def detect_and_crop(img, img_path):
    """RetinaFace detects face → crops it → returns crop + coords"""
    faces = RetinaFace.detect_faces(img_path)
    if not isinstance(faces, dict) or len(faces) == 0:
        return None, None, None

    # Pick face with highest confidence
    best_face  = max(faces.values(), key=lambda f: f["score"])
    x1, y1, x2, y2 = best_face["facial_area"]

    # Clamp coordinates to image boundaries
    h, w = img.shape[:2]
    x1   = max(0, x1)
    y1   = max(0, y1)
    x2   = min(w, x2)
    y2   = min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None

    return crop, best_face["score"], (x1, y1, x2, y2)


def get_embedding(crop):
    """ArcFace converts 112x112 crop → 512-dim embedding"""
    crop_rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (112, 112))
    return arcface.get_feat(crop_resized)[0]


def verify(embedding):
    """Compare embedding against all gallery embeddings"""
    scores = {}
    for person, gallery_emb in gallery_embeddings.items():
        sim = cosine_similarity([embedding], [gallery_emb])[0][0]
        scores[person] = round(float(sim), 4)

    best_person = max(scores, key=scores.get)
    best_score  = scores[best_person]

    if best_score >= THRESHOLD:
        result = "VERIFIED"
        identity = best_person
    else:
        result = "UNKNOWN PERSON"
        identity = "Unknown"

    return scores, best_person, best_score, result, identity


def apply_blur(img, level):
    """Gaussian blur — level 1=mild, 2=medium, 3=severe"""
    kernel_sizes = {1: 3, 2: 7, 3: 15}
    k = kernel_sizes[level]
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_compression(img, level):
    """JPEG compression — level 1=90%, 2=50%, 3=10%"""
    qualities    = {1: 90, 2: 50, 3: 10}
    q            = qualities[level]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, encoded   = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encoded, 1)


def apply_resolution(img, level):
    """Downscale then upscale — level 1=0.5x, 2=0.25x, 3=0.1x"""
    scales = {1: 0.5, 2: 0.25, 3: 0.1}
    s      = scales[level]
    h, w   = img.shape[:2]
    small  = cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s))))
    return cv2.resize(small, (w, h))


def show_scores_bar(scores, best_person, best_score, result, title):
    """Show horizontal bar chart of cosine similarity scores"""
    persons = list(scores.keys())
    values  = list(scores.values())
    colors  = ['green' if p == best_person else 'steelblue'
               for p in persons]

    plt.barh(persons, values, color=colors)
    plt.axvline(x=THRESHOLD, color='red',
                linestyle='--', label=f'Threshold ({THRESHOLD})')
    plt.xlim(0, 1)
    plt.xlabel('Cosine Similarity Score')
    plt.title(f'{title}\n{result} → {best_person.upper()} '
              f'({best_score:.4f})')
    plt.legend()

    for i, (person, val) in enumerate(zip(persons, values)):
        plt.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9)


# ============================================================
# STEP 5 — CLEAN PHOTO TEST
# ============================================================
print("\n" + "="*55)
print("CLEAN PHOTO TEST")
print("="*55)

if not os.path.exists(PROBE_PATH):
    print(f"File not found: {PROBE_PATH}")
    exit()

img        = cv2.imread(PROBE_PATH)
crop, det_conf, coords = detect_and_crop(img, PROBE_PATH)

if crop is None:
    print("No face detected in probe image!")
    exit()

x1, y1, x2, y2 = coords
print(f"Detection Confidence: {det_conf:.4f}")

# Get embedding and verify
embedding                              = get_embedding(crop)
scores, best_person, best_score, result, identity = verify(embedding)

# Print scores
print("\nCosine Similarity Scores:")
print("-" * 35)
for person, score in scores.items():
    marker = " ← BEST" if person == best_person else ""
    print(f"  {person:<15}: {score:.4f}{marker}")

print(f"\nBest Match    : {best_person.upper()}")
print(f"Best Score    : {best_score:.4f}")
print(f"Threshold     : {THRESHOLD}")
print(f"Result        : {result}")
print(f"Identity      : {identity.upper()}")

# Show clean image + scores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Clean Probe\n{result} → {identity.upper()}\n'
                  f'Score: {best_score:.4f}')
axes[0].axis('off')

plt.sca(axes[1])
show_scores_bar(scores, best_person, best_score,
                result, 'Clean Image Similarity Scores')

plt.suptitle('Clean Photo Verification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================
# STEP 6 — EXPERIMENTS 1 & 2
# Experiment 1: Accuracy under degradation (image grid)
# Experiment 2: Calibration (similarity score vs degradation)
# ============================================================
print("\n" + "="*55)
print("EXPERIMENTS 1 & 2 — DEGRADATION TESTING")
print("="*55)

degradation_types = {
    'Blur':        apply_blur,
    'Compression': apply_compression,
    'Resolution':  apply_resolution
}
level_names = {1: 'Mild', 2: 'Medium', 3: 'Severe'}

# Store results
all_scores    = {'Clean': best_score}
all_results   = {'Clean': result}
all_identities = {'Clean': identity}
all_crops     = {'Clean': crop}
all_det_fails = []

# Apply all 9 degradations
for deg_name, deg_func in degradation_types.items():
    for level in [1, 2, 3]:

        # Degrade FULL image (real world: camera captures bad image)
        img_deg   = deg_func(img, level)
        temp_path = 'temp_deg.jpg'
        cv2.imwrite(temp_path, img_deg)

        # RetinaFace on degraded image
        crop_deg, det_conf_deg, coords_deg = detect_and_crop(
            img_deg, temp_path)

        key = f'{deg_name} {level_names[level]}'

        if crop_deg is None:
            print(f"  RetinaFace failed: {key} — using original coords")
            crop_deg = img_deg[y1:y2, x1:x2]
            all_det_fails.append(key)

        # Get embedding and verify
        emb_deg                                      = get_embedding(crop_deg)
        scores_deg, bp_deg, bs_deg, res_deg, id_deg  = verify(emb_deg)

        all_scores[key]     = bs_deg
        all_results[key]    = res_deg
        all_identities[key] = id_deg
        all_crops[key]      = crop_deg

        correct = '✓' if id_deg == identity else '✗'
        print(f"  {key:<25} Score: {bs_deg:.4f} "
              f"→ {id_deg.upper():<10} {res_deg} {correct}")

# Clean up temp
if os.path.exists('temp_deg.jpg'):
    os.remove('temp_deg.jpg')

# ── EXPERIMENT 1 — Image Grid ──
print("\nGenerating Experiment 1 — Image Grid...")
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes      = axes.flatten()

for idx, (key, crop_img) in enumerate(all_crops.items()):
    id_   = all_identities[key]
    score = all_scores[key]
    res   = all_results[key]
    correct = '✓' if id_ == identity else '✗'
    color   = 'green' if id_ == identity else 'red'

    axes[idx].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(
        f'{key}\n{id_.upper()} {correct}\nScore: {score:.4f}\n{res}',
        fontsize=8,
        color=color)
    axes[idx].axis('off')

plt.suptitle(
    f'Experiment 1 — Closed Set Accuracy Under Degradation\n'
    f'Person: {identity.upper()}',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── EXPERIMENT 2 — Calibration Line Graphs ──
print("Generating Experiment 2 — Calibration Graphs...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax_idx, (deg_name, color) in enumerate(
        zip(['Blur', 'Compression', 'Resolution'],
            ['blue', 'green', 'red'])):

    scores_line = [
        all_scores['Clean'],
        all_scores[f'{deg_name} Mild'],
        all_scores[f'{deg_name} Medium'],
        all_scores[f'{deg_name} Severe']
    ]
    ids_line = [
        all_identities['Clean'],
        all_identities[f'{deg_name} Mild'],
        all_identities[f'{deg_name} Medium'],
        all_identities[f'{deg_name} Severe']
    ]
    results_line = [
        all_results['Clean'],
        all_results[f'{deg_name} Mild'],
        all_results[f'{deg_name} Medium'],
        all_results[f'{deg_name} Severe']
    ]

    axes[ax_idx].plot(
        ['Clean', 'Mild', 'Medium', 'Severe'],
        scores_line,
        f'{color[0]}-o',
        linewidth=2, markersize=8)

    # Threshold line
    axes[ax_idx].axhline(
        y=THRESHOLD, color='red',
        linestyle='--', label=f'Threshold ({THRESHOLD})')

    # Annotate each point
    for i, (score, id_, res) in enumerate(
            zip(scores_line, ids_line, results_line)):
        correct = '✓' if id_ == identity else '✗'
        axes[ax_idx].annotate(
            f'{score:.4f}\n{id_.upper()}{correct}',
            (i, score),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=8)

    axes[ax_idx].set_title(f'{deg_name} Degradation\nScore vs Level')
    axes[ax_idx].set_ylabel('Cosine Similarity Score')
    axes[ax_idx].set_ylim(0, 1.2)
    axes[ax_idx].legend()

plt.suptitle(
    f'Experiment 2 — Calibration: {identity.upper()} under Degradation\n'
    f'Does similarity score drop honestly as degradation increases?',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Summary Table ──
print(f"\n{'Type':<25} {'Score':>8} {'Identity':>12} "
      f"{'Result':>15} {'Correct?':>10}")
print("-" * 75)
for k, v in all_scores.items():
    id_     = all_identities[k]
    res     = all_results[k]
    correct = '✅' if id_ == identity else '❌'
    print(f"{k:<25} {v:>8.4f} {id_.upper():>12} "
          f"{res:>15} {correct:>10}")

# Detection failures summary
if all_det_fails:
    print(f"\nRetinaFace failed on {len(all_det_fails)} degradations:")
    for f in all_det_fails:
        print(f"  → {f}")

print("\nDone!")
