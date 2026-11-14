# Explainable AI for Diabetic Retinopathy Screening

Academic prototype for five-grade DR grading from retinal fundus photographs using ImageNet EfficientNet-B0, referable DR derivation and Grad-CAM.

> Research prototype only. Not a medical diagnosis or clinically validated product.

## Repository and data audit

The local checkout previously had a README and IDRiD files but no implementation. The handoff's selected dataset (APTOS 2019) conflicts with the actual local dataset: this checkout contains **IDRiD**, with 413 labeled training and 103 labeled official test images. This implementation uses the present IDRiD data and does not combine datasets. The local grade counts (train/test) are 0: 134/34, 1: 20/5, 2: 136/32, 3: 74/19, 4: 49/13.

Expected data layout:
```
data/B. Disease Grading/
├── 1. Original Images/{a. Training Set,b. Testing Set}/
└── 2. Groundtruths/{a. IDRiD_Disease Grading_Training Labels.csv,b. IDRiD_Disease Grading_Testing Labels.csv}
```
The code checks label/image matches, removes exact duplicate content, and records the affected IDs. A fixed-seed stratified validation split comes only from official training; the official test partition is used only for final scoring. One duplicated official test image was excluded, leaving 102 unique test images. Review the [official IDRiD page](https://idrid.grand-challenge.org/Data/) and cite IDRiD in academic work. Dataset license terms are separate from code licensing.

## Setup

Use Python 3.10–3.12 and install a matching PyTorch/torchvision build from the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then:
```powershell
python -m pip install -r requirements.txt
```
The model was trained in the inspected Python 3.13.7 / PyTorch 2.8.0 CPU environment after installing matching torchvision 0.23.0. There was no CUDA GPU, so training ran on CPU.

## Train and evaluate

```powershell
python -m src.pipeline train
python demo.py --accuracy
```
Config defaults: 224px, ImageNet normalization, conservative train-only flips/rotation, inverse-square-root weighted cross entropy, AdamW, seed 42 and early stopping by validation loss. Pretrained weights download on first training run. Use `--no-pretrained` to skip that download.

The post-training demo prints accuracy, per-class precision/recall/F1, macro-F1, quadratic weighted kappa, and referable sensitivity/specificity. It writes `reports/test_metrics.json`, `reports/test_predictions.csv` and `reports/test_confusion_matrix.png` (these generated files remain local and are ignored by Git).

## Measured baseline

Early stopping selected epoch 7 by validation loss (0.8997). On the deduplicated official test set (n=102), the measured metrics were accuracy **0.510**, macro-F1 **0.410**, quadratic weighted kappa **0.656**, referable sensitivity **0.873**, and referable specificity **0.769**. Mild DR recall was **0/5**. These small-sample results show substantial limitations and must not be presented as clinical performance.

## Related projects and what we borrowed

These figures are context, not a head-to-head benchmark: test-set handling, duplicate removal, architectures, training protocols, and metrics are not identical.

| Project | IDRiD result or method | Comparison |
|---|---|---|
| This project | Official test, 102 unique images: accuracy 0.510, macro-F1 0.410, QWK 0.656 | EfficientNet-B0 transfer-learning baseline |
| [DeepakSkandh's IDRiD feature-classification project](https://github.com/DeepakSkandh/diabetic-retinopathy-feature-classification) | Reports a 103-image test result for its v2 SVM: accuracy 0.388, macro-F1 0.364, QWK 0.590; its nested-CV accuracy is around 0.49–0.55 | Our single test estimate is higher on these metrics, but our duplicate-cleaned set has 102 images and their method is classical features + SVM |
| [nnMobileNet (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/papers/Zhu_nnMobileNet_Rethinking_CNN_for_Retinopathy_Research_CVPRW_2024_paper.pdf) | Reports IDRiD five-grade accuracy 0.731 and AUC 0.916 using the official split | A substantial benchmark gap: our accuracy is 22.1 percentage points lower; the custom architecture and evaluation details differ |

The IDRiD project above emphasizes nested cross-validation, bootstrap intervals, class-wise metrics, and reporting test results cautiously. We added bootstrap uncertainty and kept loss comparisons on validation data. The [APTOS loss-ablation project](https://github.com/mshaiel/dr-severity-grading) compares weighted cross-entropy, focal loss, and CORAL; because it uses a much larger, different dataset, we borrowed the controlled-ablation idea only and did not compare its scores with ours.

### Validation-only class-weighting ablation

Using the same 80-image validation partition, seed, pretrained backbone, and training setup, changing only whether inverse-square-root class weights are applied gave:

| Loss | Accuracy | Macro-F1 | QWK | Mild recall (3 cases) |
|---|---:|---:|---:|---:|
| Weighted cross-entropy | 0.663 | 0.508 | 0.794 | 0/3 |
| Unweighted cross-entropy | 0.638 | 0.566 | 0.784 | 1/3 |

The weighted run scored slightly higher on accuracy/QWK; the unweighted run scored higher on macro-F1 and found one validation Mild case. Three Mild validation examples are far too few to establish a winner. This ablation is reported as a trade-off, not as evidence that either weighting strategy is generally better. It did not use the official test set.

Reproduce it with `python scripts/run_loss_ablation.py`. It saves the comparison to `reports/loss_weighting_ablation.csv` and a chart; the unweighted checkpoint is local under `artifacts/checkpoints/`.

### Test-set uncertainty

We also computed a fixed-seed, 5,000-replicate **stratified paired bootstrap** from the already-saved test predictions (resampling within each true grade). The 95% percentile intervals are accuracy **0.422–0.598**, macro-F1 **0.327–0.491**, QWK **0.533–0.763**, referable sensitivity **0.794–0.952**, and specificity **0.641–0.897**. These intervals show how uncertain the 102-image estimate is; they do not measure cross-hospital generalization.

Recreate the interval file without rerunning inference with `python scripts/bootstrap_test_metrics.py`. The summary is in `reports/test_bootstrap_ci.json`.

Single-image inference and Grad-CAM:
```powershell
python demo.py --image "path\to\fundus.jpg"
```
Grad-CAM targets EfficientNet-B0's final feature block and indicates gradient sensitivity; it does not establish clinical reasoning.

## Optional Streamlit UI

The command-line demo covers prediction and visualization. Run this UI after training to review fixed-scale training charts, screen one image locally, inspect Grad-CAM, and download a one-page PDF report:
```powershell
streamlit run app/app.py
```

## GitHub cross-check

Reviewed [danort92/diabetic-retinopathy-grading](https://github.com/danort92/diabetic-retinopathy-grading) as a reference for stratified splits, duplicate checks, per-class evaluation and Grad-CAM at the final EfficientNet convolutional block. That project uses APTOS-specific CORAL/preprocessing; none of its reported results are attributed to this IDRiD baseline. The implementation follows the project's already chosen five-class cross-entropy baseline instead.
