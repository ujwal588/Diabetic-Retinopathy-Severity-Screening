# Diabetic Retinopathy Severity Screening

A supervised learning research project exploring how transfer-learning CNNs can classify diabetic retinopathy (DR) severity from retinal fundus photographs. The aim is to study whether image-based predictions can help prioritize potentially high-risk cases for specialist review.

> **Research scope:** This project is for education and research. Model outputs are not a medical diagnosis, should not guide treatment, and must not replace review by a qualified eye-care professional.

## Task

Predict one of five ordered DR severity grades from a color fundus image:

| Grade | Severity |
|---:|---|
| 0 | No DR |
| 1 | Mild non-proliferative DR |
| 2 | Moderate non-proliferative DR |
| 3 | Severe non-proliferative DR |
| 4 | Proliferative DR |

The planned experiments focus on transfer learning, class imbalance, per-class sensitivity/recall, confusion-matrix analysis, and Grad-CAM visualizations.

## Dataset

The project currently uses the **Indian Diabetic Retinopathy Image Dataset (IDRiD)**. The downloaded IDRiD packages are stored locally under `data/`:

- `data/B. Disease Grading/` — image-level DR grades; primary package for the classification task.
- `data/A. Segmentation/` — retinal images and lesion segmentation ground truths; optional for lesion-focused analysis.
- `data/C. Localization/` — retinal images and localization ground truths; optional for localization experiments.

The official IDRiD page describes 516 image-level graded fundus images and provides the dataset with a CC BY 4.0 license. Review and follow the included `LICENSE.txt` and `CC-BY-4.0.txt` notices when using or sharing the data. Cite the dataset in any report or presentation. See the [official IDRiD dataset page](https://idrid.grand-challenge.org/Data/).

### Keep data out of Git

The image files and labels are excluded by `.gitignore` because they are large dataset assets and should not be committed to this repository. Download the dataset from its official source and extract the three packages into `data/` using the folder names above. Do not remove the ignore rule to force-add the dataset. If another contributor needs the data, share the official source and setup instructions rather than a copy of the files.

## Planned workflow

1. Inspect labels, image quality, class counts, and package contents.
2. Use IDRiD's provided training/testing label files: keep its 103-image testing split untouched and create a stratified validation split from the 413-image training partition before augmentation. Keep patient-level groups together if identifiers are available.
3. Preprocess fundus photographs consistently and apply augmentation only to training examples.
4. Fine-tune a pretrained CNN and compare class-imbalance strategies such as weighted loss or balanced sampling.
5. Select model settings using validation data, then evaluate once on the held-out test split.
6. Report per-class recall/sensitivity, precision, F1, support, macro-F1, and a confusion matrix; include accuracy only as a supplementary metric.
7. Review Grad-CAM examples across grades, including errors, and document limitations and dataset shift.

No model performance claims are made until these experiments have been run and reported.

## Repository status

This repository currently contains the dataset folder and this project overview. Training code, notebooks, evaluation outputs, and Grad-CAM examples can be added as the implementation develops. A suitable structure is:

```text
.
├── data/          # Local only; ignored by Git
├── notebooks/     # Exploration and experiments
├── src/           # Reusable preprocessing, training, and evaluation code
├── reports/       # Figures and written results (no raw images)
├── README.md
└── .gitignore
```

## License and attribution

The project code and the IDRiD dataset have separate licensing terms. Follow the license files distributed with IDRiD and the official dataset page. Add an explicit license for this repository's code before redistribution; do not imply that the dataset is included here.
