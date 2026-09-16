# AI-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory

## 1. Project Overview

This project develops an AI-powered soil analytics system that combines:

1. **Soil image classification** using a CNN with transfer learning.
2. **Structured soil-data machine learning** for N, P, and K nutrient-deficiency assessment.
3. **Explainable AI (XAI)** using SHAP and Grad-CAM.
4. **Hybrid soil assessment** combining soil-type information from images with nutrient-status predictions from structured data.

The project is organized as a milestone-based workflow covering data preparation, exploratory data analysis, model development, evaluation, explainability, and hybrid integration.

---

## 2. Objectives

- Classify soil images into seven soil-type classes.
- Develop nutrient-deficiency prediction models for Nitrogen (N), Phosphorus (P), and Potassium (K).
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Explain structured-model predictions using SHAP.
- Visualize important image regions using Grad-CAM.
- Combine image-based soil classification and structured nutrient predictions into a single assessment.
- Save trained models and generated outputs for reuse and reporting.

---

## 3. Dataset

### Image Dataset

The cleaned image dataset contains:

- **1,139 images**
- **7 soil classes**
- RGB images standardized to **224 × 224**
- Duplicate removal was performed before model development.
- **49 duplicate images** were removed from the original collection.

Soil classes:

- Alluvial Soil
- Arid Soil
- Black Soil
- Laterite Soil
- Mountain Soil
- Red Soil
- Yellow Soil

The final image split contains:

| Split | Images |
|---|---:|
| Training | 793 |
| Validation | 168 |
| Test | 178 |
| **Total** | **1,139** |

The image dataset is imbalanced, with Arid Soil having the largest representation and Alluvial Soil the smallest.

### Structured Dataset

The structured soil dataset contains **2,200 records and 23 columns**.

It includes soil nutrients and environmental/agricultural variables such as:

- N, P, K
- Temperature
- Humidity
- pH
- Rainfall
- Soil moisture
- Soil type
- Sunlight exposure
- Wind speed
- CO₂ concentration
- Organic matter
- Irrigation frequency
- Crop density
- Pest pressure
- Fertilizer usage
- Growth stage
- Urban-area proximity
- Water-source type
- Frost risk
- Water-use efficiency

The original `label` column represents **crop recommendation** and does not provide laboratory-confirmed N/P/K deficiency labels.

Therefore, the structured ML stage derives **dataset-relative proxy deficiency labels** using the 25th percentile of each nutrient:

| Nutrient | Threshold | Deficient records |
|---|---:|---:|
| N | 21 | 536 |
| P | 28 | 540 |
| K | 20 | 478 |

These are **proxy labels for this project**, not agronomic or laboratory-confirmed deficiency measurements.

---

## 4. System Architecture

```text
                    ┌──────────────────────┐
                    │     Soil Image       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ EfficientNetB0 CNN   │
                    │ Transfer Learning    │
                    └──────────┬───────────┘
                               │
                               ▼
                       Soil Type + Confidence
                               │
                               │
                               │       ┌────────────────────────┐
                               │       │ Structured Soil Data    │
                               │       └───────────┬────────────┘
                               │                   │
                               │                   ▼
                               │       ┌────────────────────────┐
                               │       │ Gradient Boosting       │
                               │       │ N / P / K Models        │
                               │       └───────────┬────────────┘
                               │                   │
                               │                   ▼
                               │       N/P/K Status + Confidence
                               │                   │
                               └──────────┬────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ Hybrid Assessment    │
                              │ + Nutrient Score     │
                              └──────────────────────┘

                  Explainability:
                  • SHAP → structured features
                  • Grad-CAM → image regions
```

The hybrid stage uses **late assessment-level fusion** rather than averaging model probabilities because the image model and structured models predict different types of outputs.

---

## 5. CNN Soil Image Classification

### Model Selection

Two common transfer-learning choices were considered:

- ResNet-50
- EfficientNetB0

**EfficientNetB0 was selected** because the project uses a relatively small image dataset and requires a computationally efficient transfer-learning model. This does not mean EfficientNetB0 is universally better than ResNet-50; the choice is specific to this project.

### Transfer Learning

The CNN uses ImageNet-pretrained EfficientNetB0 features.

Initial training:

- Base model frozen
- Global Average Pooling
- Dropout = 0.3
- Dense output layer = 7 classes
- Adam optimizer
- Learning rate = 0.001
- Sparse categorical cross-entropy
- Early stopping and model checkpointing

### Fine-Tuning

After initial transfer learning, the upper part of EfficientNetB0 was fine-tuned:

- First 200 of 238 EfficientNetB0 layers remained frozen.
- Upper 38 layers were trainable.
- Learning rate was reduced to **1e-5**.

This allowed the pretrained representation to adapt more gently to the soil-image domain.

### CNN Results

| Model stage | Test Accuracy |
|---|---:|
| Initial transfer learning | **90.45%** |
| Fine-tuned EfficientNetB0 | **92.13%** |

Fine-tuning improved test accuracy by **1.68 percentage points**.

The final fine-tuned model was selected and saved as:

`models/efficientnet_soil_classifier_finetuned.keras`

### Final Test Classification Report

| Soil Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Alluvial | 0.67 | 0.25 | 0.36 | 8 |
| Arid | 0.91 | 0.93 | 0.92 | 43 |
| Black | 0.97 | 0.97 | 0.97 | 36 |
| Laterite | 0.88 | 0.91 | 0.90 | 33 |
| Mountain | 0.91 | 0.97 | 0.94 | 30 |
| Red | 1.00 | 1.00 | 1.00 | 17 |
| Yellow | 0.92 | 1.00 | 0.96 | 11 |

Overall test accuracy: **92.13%**

The final confusion matrix contains **164 correct predictions out of 178 test images**.

A notable limitation is the relatively low recall for the Alluvial Soil class, which has the smallest test support.

---

## 6. Structured Soil Machine Learning

Gradient Boosting Classifiers were trained separately for:

- Nitrogen deficiency
- Phosphorus deficiency
- Potassium deficiency

The nutrient used to construct each proxy target was excluded from that target's input features to reduce direct target leakage.

For example:

- N model excludes N.
- P model excludes P.
- K model excludes K.

The dataset was split using an **80/20 stratified train/test split** with `random_state=42`.

Tree-based Gradient Boosting models were used without feature scaling.

### Model Evaluation

The primary metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

F1-score was used as the tuning objective because the deficiency classes are not perfectly balanced and both precision and recall are important.

### Final Models

| Nutrient | Model retained | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| N | Tuned Gradient Boosting | 0.7500 | 0.4815 | 0.3645 | 0.4149 |
| P | Baseline Gradient Boosting | 0.9273 | 0.8333 | 0.8796 | 0.8559 |
| K | Tuned Gradient Boosting | 0.8091 | 0.5857 | 0.4271 | 0.4940 |

Hyperparameter tuning used 5-fold cross-validation over:

- `n_estimators`: 100, 150, 200
- `learning_rate`: 0.03, 0.05, 0.1
- `max_depth`: 2, 3, 4
- `min_samples_split`: 2, 5, 10

The tuned model was not automatically retained for every nutrient. For P, the baseline model performed slightly better on the test set, so the baseline model was retained.

Saved models:

```text
Models/
├── N_deficiency_gradient_boosting.pkl
├── P_deficiency_gradient_boosting.pkl
└── K_deficiency_gradient_boosting.pkl
```

---

## 7. SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the Gradient Boosting models.

`TreeExplainer` was used because the models are tree-based Gradient Boosting classifiers.

SHAP was used at two levels:

### Global Explanation

Global SHAP visualizations show which structured features have the largest overall contribution to model predictions.

Examples of frequently important features include:

- Humidity
- K
- Rainfall
- P
- N
- Temperature
- pH
- Fertilizer usage
- Water-use efficiency

The exact feature importance differs by nutrient model.

### Local Explanation

Waterfall plots were used to explain individual predictions.

A positive SHAP contribution moves the prediction toward the explained class, while a negative contribution moves it away from that class.

SHAP explanations describe **model behavior**. They should not be interpreted as proof of causal agronomic relationships.

---

## 8. Grad-CAM Explainability

Grad-CAM was applied to the final EfficientNetB0 model to visualize image regions contributing to a prediction.

The final convolutional feature layer used for Grad-CAM is:

`top_activation`

Its feature-map output is:

`7 × 7 × 1280`

Grad-CAM examples include:

- A correct Alluvial Soil prediction.
- An incorrect Alluvial Soil prediction classified as Laterite Soil.

For the incorrect example, the model predicted Laterite Soil with approximately **47.8% confidence**.

For the correct example, the model predicted Alluvial Soil with approximately **73.0% confidence**.

The heatmaps provide an indication of where the model's visual representation was concentrated. They are **not causal explanations** and do not prove that the highlighted soil region is the scientifically relevant cause of the prediction.

---

## 9. Hybrid Soil Assessment

The hybrid notebook integrates:

### Image model

Outputs:

- Soil type
- Image confidence

### Structured models

Outputs:

- N deficiency status + confidence
- P deficiency status + confidence
- K deficiency status + confidence

### Nutrient-status score

The project defines a simple nutrient-status score:

- 3/3 nutrients not deficient → **100**
- 2/3 → **66.67**
- 1/3 → **33.33**
- 0/3 → **0**

This is a **project-defined nutrient-status score**, not a scientifically validated soil-health score.

Example hybrid outputs include:

| Sample | Soil Type | N | P | K | Score |
|---|---|---|---|---|---:|
| 0 | Alluvial | Not Deficient | Not Deficient | Not Deficient | 100 |
| 500 | Alluvial | Deficient | Not Deficient | Deficient | 33.33 |
| 1000 | Alluvial | Not Deficient | Not Deficient | Not Deficient | 100 |
| 1500 | Alluvial | Not Deficient | Not Deficient | Not Deficient | 100 |
| 2000 | Alluvial | Not Deficient | Not Deficient | Not Deficient | 100 |

The hybrid demonstration uses a fixed test image together with selected structured records. There are **no verified image-to-structured physical soil-sample pairings** in the current dataset. Therefore, these examples demonstrate system integration rather than paired-sample validation.

---

## 10. Explainability and Assessment Outputs

Generated outputs are stored under:

```text
Outputs/
├── CNN/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── SHAP/
│   ├── N_SHAP_summary.png
│   ├── P_SHAP_summary.png
│   ├── K_SHAP_summary.png
│   ├── N_SHAP_bar.png
│   ├── P_SHAP_bar.png
│   └── K_SHAP_bar.png
├── GradCAM/
│   ├── correct_prediction.png
│   └── incorrect_prediction.png
└── Hybrid/
    └── hybrid_results.csv
```

---

## 11. Notebook Workflow

The notebooks are organized in the following order:

| Notebook | Purpose |
|---|---|
| `01_Project_Setup.ipynb` | Project setup and environment |
| `02_Structured_Data_EDA.ipynb` | Structured dataset exploration |
| `03_Image_Data_EDA.ipynb` | Image dataset exploration and preprocessing |
| `04_CNN_Soil_Image_Classification.ipynb` | EfficientNetB0 training and evaluation |
| `05_Structured_Soil_ML.ipynb` | N/P/K Gradient Boosting models |
| `06_SHAP_Explainability.ipynb` | SHAP global and local explanations |
| `07_GradCAM_Explainability.ipynb` | CNN visual explanations |
| `08_Hybrid_Soil_Analysis.ipynb` | Integrated soil assessment |

---

## 12. Project Structure

```text
Soil_Analytics_Project/
├── Data/
│   ├── Metadata/
│   ├── Processed/
│   │   ├── crops_npk/
│   │   ├── EarlyNSD/
│   │   ├── Image_Data/
│   │   └── Image_Split/
│   └── Raw/
│       ├── Image_Data/
│       └── Structured_Data/
│
├── Models/
│   ├── efficientnet_soil_classifier_finetuned.keras
│   ├── efficientnet_soil_classifier.keras
│   ├── N_deficiency_gradient_boosting.pkl
│   ├── P_deficiency_gradient_boosting.pkl
│   └── K_deficiency_gradient_boosting.pkl
│
├── Notebooks/
│   ├── 01_Project_Setup.ipynb
│   ├── 02_Structured_Data_EDA.ipynb
│   ├── 03_Image_Data_EDA.ipynb
│   ├── 04_CNN_Soil_Image_Classification.ipynb
│   ├── 05_Structured_Soil_ML.ipynb
│   ├── 06_SHAP_Explainability.ipynb
│   ├── 07_GradCAM_Explainability.ipynb
│   └── 08_Hybrid_Soil_Analysis.ipynb
│
├── Outputs/
│   ├── CNN/
│   ├── SHAP/
│   ├── GradCAM/
│   └── Hybrid/
│
└── Reports/
    ├── Dataset_Documentation_Report.docx
    └── Dataset_License_and_Usage_Documentation.md
```

---

## 13. How to Use

The notebooks were developed as a sequential workflow.

Recommended order:

1. Set up the project and paths.
2. Review structured-data EDA.
3. Review image-data EDA and preprocessing.
4. Train/evaluate or inspect the saved CNN model.
5. Train/evaluate or inspect the saved structured ML models.
6. Run SHAP explanations.
7. Run Grad-CAM explanations.
8. Run the hybrid assessment.

The notebooks use project paths under the project directory. When running in a different environment, update the project root/path variables to match the local or Google Drive location.

Saved models can be loaded directly from the `Models/` directory for inference and explanation without retraining.

---

## 14. Important Limitations

### 1. Proxy nutrient labels

The N/P/K deficiency targets are derived from dataset-relative percentile thresholds. They are not laboratory-confirmed deficiency labels.

### 2. Image and structured data are not physically paired

The current hybrid demonstration does not validate predictions on the same physical soil sample using both image and structured measurements.

### 3. Project-defined soil-health score

The 0–100 nutrient-status score is a simple project-defined rule based on the number of modeled deficiencies. It is not a scientifically validated soil-health index.

### 4. Class imbalance

The image dataset contains substantially different numbers of examples across soil classes. In particular, Alluvial Soil has relatively low support, which is reflected in its lower recall.

### 5. Model explanations

SHAP and Grad-CAM explain model behavior and attention patterns. They should not be treated as causal scientific evidence.

---

## 15. Key Results at a Glance

| Component | Final Result |
|---|---|
| Soil image classifier | EfficientNetB0 |
| Image classes | 7 |
| Image dataset | 1,139 images |
| CNN test accuracy | **92.13%** |
| N model | Tuned Gradient Boosting |
| N F1 | **0.4149** |
| P model | Baseline Gradient Boosting |
| P F1 | **0.8559** |
| K model | Tuned Gradient Boosting |
| K F1 | **0.4940** |
| Structured XAI | SHAP |
| Image XAI | Grad-CAM |
| Integration | Hybrid late assessment-level fusion |

---

## 16. Milestone 2 Deliverables

The completed Milestone 2 workflow provides:

- Trained CNN soil-image classifier.
- CNN evaluation metrics and confusion matrix.
- Structured N/P/K Gradient Boosting models.
- Hyperparameter tuning and model-selection results.
- SHAP global and local explanations.
- Grad-CAM visual explanations.
- Hybrid soil assessment function and example results.
- Saved model files.
- Generated plots and CSV outputs.
- Project documentation and final report.

