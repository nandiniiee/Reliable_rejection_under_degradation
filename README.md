# RELIABLE REJECTION UNDER IMAGE DEGRADATION USING QUALITY AWARE VERIFICATION FRAMEWORK
A lightweight quality-aware face verification framework that improves verification reliability under degraded imaging conditions by combining **ArcFace embeddings**, **MUSIQ image quality assessment**, and a **Multi-Layer Perceptron (MLP)**.

Unlike traditional verification systems that rely on a fixed cosine similarity threshold, this project learns verification confidence from multiple features, resulting in significantly improved performance on degraded, natural, and low-quality face images.

## MOTIVATION
Most modern face recognition systems perform verification using a simple pipeline:

```
Face Image
↓
ArcFace
↓
Embedding
↓
Cosine Similarity
↓
Fixed Threshold
↓
Accept / Reject
```
Although effective under controlled conditions, fixed-threshold verification often fails when images suffer from:
  - Motion blur
  - Low resolution
  - Poor illumination
  - Compression artifacts
  - Natural image degradation

A degraded image may produce the same cosine similarity as a high-quality image, even though the confidence of the prediction is much lower.

This project addresses that limitation by incorporating **image quality** and **similarity margin** into the final decision.

## PROPOSED FRAMEWORK
Instead of relying only on cosine similarity, the proposed framework combines three complementary features:

- **Cosine Similarity** (ArcFace)
- **Similarity Margin** (difference between the best similarity score and second-best similarity score)
- **MUSIQ Quality Score** (perceptual image quality)

These features are provided to a lightweight Multi-Layer Perceptron (MLP), which predicts whether a verification attempt should be **accepted** or **rejected**.

```
Gallery Images
│
├── ArcFace Embeddings
│
└── Gallery Database
↓
Probe Image
↓
ArcFace Embedding
↓
MUSIQ Quality Score
↓
Cosine Similarity Search
↓
Best Similarity Score
↓
Margin (Best Score - Second Best Score)
↓
Feature Vector (Gives 3-D Vector)
↓
MLP Verification Model
↓
Accept / Reject
↓
Predicted Identity
```

## FEATURES
- ArcFace-based face embeddings
- MUSIQ no-reference image quality assessment
- Similarity margin estimation
- Lightweight MLP verification classifier
- Extensive ablation study
- TinyFace evaluation
- Cross-demographic evaluation (African and Asian Datasets)
- Unknown identity testing
- Confusion matrix generation

## REPOSITORY STRUCTURE
```
Reliable-Rejection-Under-Degradation/
│
├── verification_mlp/multiLayer/mlp_multiLayer_fullExtended
│ ├── clean/
│ ├── degraded/
│ ├── natural/
│ ├── unknown/
│ ├── tinyface/
│ └── african_dataset/
│ ├── asian_dataset/
│ ├── train_mlp/
│ ├── test_mlp/
│ └── comapre_magFace/
│ ├── compare_vggNet/
│ ├── save_model/
│ ├── save_scaler/
│ └── embedding generation/
│ ├── faceCropping/
│ ├── csv_generation/

├── far_test_calculation/
│ ├── closed_set_calibration/
│ └── open_set_calibration/
│

├── ablation_study/
│ ├── test_A1_sim/
│ ├── test_A2_simMarg/
│ ├── test_A3_simQuali/
│ ├── test_A4_qualiMarg/
│ └── test_A5_all/
│

├── verification_svm/
│ ├── clean/
│ ├── degraded/
│ ├── unknown/
│ ├── tinyface/
│ └── dataset_split_creation/
│ ├── lfw_dataset/
│ ├── train_svm/
│ ├── test_svm/
│ └── impostor_testing/
│ ├── hardNegatives_generation/
│ ├── save_model/
│ ├── save_scaler/
│ └── embedding generation/
│ ├── faceCropping/
│ ├─

├── figures/
│

├── plotCreation/
│

└── README.md
```

## METHODOLOGY

### 1. FACE DETECTION
Faces are represented using **ArcFace (Buffalo-L)** embeddings.

### 2. IMAGE QUALITY ASSESSMENT
Probe image quality is estimated using the **MUSIQ** no-reference image quality assessment model.

### 3. SIMILARITY SEARCH
Each probe embedding is compared against every gallery embedding using cosine similarity.

For each probe, the framework computes: 
   - Best similarity
   - Second-best similarity
   - Similarity margin

### 4.VERIFICATION
The final feature vector: 

*[Similarity, Margin, Quality Score]*

is passed to a Multi-Layer Perceptron, which predicts whether the verification should be accepted or rejected.

## DATASETS USED
The framework was evaluated on multiple datasets covering different verification scenarios.
|_________________|__________________________________|
|     Dataset     |            Purpose               |
|_________________|__________________________________|
|      Clean      | Baseline verification            |
|_________________|__________________________________|
|     Degraded    | Synthetic image degradation      |
|_________________|__________________________________|
|     Natural     | Real-world image degradation     |
|_________________|__________________________________|
|     Unknown     | Unknown identity rejection       |
|_________________|__________________________________|
|  TinyFace like  | Low-resolution face verification |
|_________________|__________________________________|
| African Dataset | Cross-demographic evaluation     |
|_________________|__________________________________|


## ABlATION STUDY
To evaluate the contribution of each feature, five different models were trained.

|_______|_______________________________|
| Model |          Features             |
|_______|_______________________________|
| A1    | Similarity                    |
|_______|_______________________________|                
| A2    | Similarity + Margin           |
|_______|_______________________________|
| A3    | Similarity + Quality          |
|_______|_______________________________|
| A4    | Margin + Quality              |
|_______|_______________________________|
| A5    | Similarity + Margin + Quality |
|_______|_______________________________|

The study demonstrates that incorporating image quality and confidence estimation substantially improves verification performance compared to similarity-only verification.

## RESULTS
The proposed framework consistently outperformed conventional fixed-threshold verification across multiple datasets.

Key observations include:
   - Improved verification under degraded images
   - Higher robustness on TinyFace like datasets
   - Better generalization across demographics
   - Reduced false rejections
   - Maintained low false acceptance rate

Example evaluation metrics include:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - True Acceptance Rate (TAR)
   - False Rejection Rate (FRR)
   - True Rejection Rate (TRR)
   - False Acceptance Rate (FAR)

## Technologies Used

- Python
- PyTorch
- ArcFace (InsightFace)
- RetinaFace
- MUSIQ (PyIQA)
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Seaborn

## CITATION
If you find this repository useful in your research, please consider citing this work.

```bibtex
@misc{reliable_rejection_under_degradation,
  title={Reliable Rejection Under Image Degradation using Quality-Aware Face Verification},
  author={Nandini Jain, Shaurya Sharma},
  year={2026}
}
```

## ACKNOWLEDGMENTS
This project builds upon the following open-source works:
   - InsightFace (ArcFace)
   - RetinaFace
   - PyIQA
   - MUSIQ
   - PyTorch
   - Scikit-learn

Special thanks to the open-source community for making these tools publicly available.