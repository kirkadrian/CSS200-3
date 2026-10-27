<div align="center">

# Performance Evaluation of Convolutional Neural Networks for E-Waste Laptop Components

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)
![Dataset License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-2E8B57?style=for-the-badge)
![Code License](https://img.shields.io/badge/Code-MIT-blue?style=for-the-badge)

</div>

---

> A thesis project by Kirk Adrian E. Sunico, Prince Jeffrey Villamil, Radzie R. Bagallon, and John Paul Q. Tomas (adviser)
> School of Information Technology, Mapúa University, Makati, Philippines

## Overview

The growing volume of electronic waste in the Philippines is made worse by the hazards and inefficiency of manual classification at Material Recovery Facilities (MRFs). This study evaluates convolutional neural networks (CNNs) for the automated classification of five laptop components: **motherboard, battery, random-access memory (RAM), WiFi card, and central processing unit (CPU) fan**.

It tests whether applying **focal loss** and **three-stage progressive unfreezing** to a baseline **DenseNet-121** gives a statistically significant improvement, and benchmarks this against a lightweight **MobileNet-V2**.

**Data.** A balanced five-class subset of 700 images (140 per class) was taken from the Mendeley Data *Laptop Components* image dataset. A Laplacian variance blur filter removed 57 images, leaving **643**. They were split at the image level, 80/10/10, into **514 training, 64 validation, and 65 test** images. Only the training partition was augmented.

**Configurations.** Four configurations were initialized from ImageNet-1k weights and each trained with three independent random seeds (Adam optimizer, batch size 16):

| Configuration | Loss function | Epochs | Backbone training | Trainable parameters |
|---|---|---|---|---|
| Baseline DenseNet-121 | Cross-entropy | 50 | Frozen | 263,685 |
| Baseline MobileNet-V2 | Cross-entropy | 50 | Frozen | 329,221 |
| Optimized DenseNet-121 (Frozen Control) | Focal loss | 100 | Frozen | 263,685 |
| Optimized DenseNet-121 (Unfreeze) | Focal loss | 100 | Progressive unfreezing from epoch 68 | 263,685 to 5,850,629 |

Focal loss used γ = 2.0 and α = 0.25.

**Evaluation.** In-distribution classification on the 65-image test set, an embedding-similarity audit for cross-split duplication, an out-of-distribution (OOD) test on 100 uncropped, researcher-collected images (50 clean white backgrounds, 50 cluttered backgrounds), and a computational efficiency benchmark. OOD differences were tested with a paired bootstrap and Holm correction (18 comparisons).

---

## Results

**In-distribution.** All four configurations classified all 65 test images correctly in every seed (accuracy, macro-precision, macro-recall, macro-F1, and macro ROC-AUC of 1.0000; exact 95% confidence interval for accuracy, 0.945 to 1.000). The test set could not separate the configurations.

**Embedding audit.** The audit flagged 52 cross-split image pairs (cosine similarity 0.87 to 0.93) in the Battery, CPU Fan, and RAM classes, mostly the same physical component photographed from slightly different angles. The perfect scores are therefore an experimental ceiling effect and are best read as upper bounds for unseen components.

**Out-of-distribution accuracy** (mean ± SD across three seeds; 100 images, 50 per background):

| Configuration | Overall accuracy (%) | White background (%) | Messy background (%) | Macro-F1 |
|---|---|---|---|---|
| Baseline MobileNet-V2 | 69.7 ± 2.5 | 75.3 ± 3.1 | 64.0 ± 2.0 | 0.673 ± 0.025 |
| Baseline DenseNet-121 | 69.3 ± 3.5 | 76.0 ± 0.0 | 62.7 ± 7.0 | 0.680 ± 0.039 |
| Optimized DenseNet-121 (Frozen Control) | 67.0 ± 1.0 | 78.7 ± 1.2 | 55.3 ± 1.2 | 0.622 ± 0.012 |
| Optimized DenseNet-121 (Unfreeze) | 71.0 ± 1.7 | 78.0 ± 2.0 | 64.0 ± 2.0 | 0.682 ± 0.025 |

Accuracy fell by roughly 29 to 33 percentage points from the test set. No Holm-adjusted difference between configurations was significant. The largest observed difference, Unfreeze over Frozen Control on messy backgrounds (+8.7 points, unadjusted p ≈ 0.06), is exploratory and not a confirmed effect.

**Computational efficiency** (NVIDIA Tesla T4 GPU):

| Configuration | Model size (MB) | Inference time (ms) | Computation (GFLOPs) |
|---|---|---|---|
| Baseline MobileNet-V2 | 10.44 | 5.59 ± 0.09 | 0.61 |
| Baseline DenseNet-121 | 29.30 | 18.11 ± 3.72 | 5.70 |
| Optimized DenseNet-121 (Frozen Control) | 29.30 | 16.46 ± 1.33 | 5.70 |
| Optimized DenseNet-121 (Unfreeze) | 29.30 | 16.16 ± 1.02 | 5.70 |

MobileNet-V2 is about 2.8 times smaller, 2.9 to 3.2 times faster, and needs 9.3 times fewer FLOPs than the DenseNet-121 configurations.

**Conclusion.** The optimizations gave no measurable in-distribution gain and no statistically supported OOD advantage. MobileNet-V2 is recommended for controlled operational environments, but no evaluated model is robust enough for heavily cluttered facility floors without physical pre-sorting and environmental standardization.

**Limitations.** The image-level split let near-duplicate views of the same component into the test set. The OOD test combines domain shift, background clutter, and the removal of the label-dependent cropping step, and its 100 images limit statistical power. Training was not seeded, so retraining will not reproduce the exact weights or metrics (inference on the saved checkpoints is deterministic).

---

## Setup (Google Colab)

All experiments were implemented in a single Google Colab notebook on an NVIDIA Tesla T4 GPU (16 GB) using TensorFlow and Keras, so no local installation is needed.

1. **Get the files.** Clone this repository or download it as a ZIP, then upload the folder to your Google Drive.
2. **Get the dataset.** Download the *Laptop Components* dataset from Mendeley Data (doi: [10.17632/pwctygncrj.4](https://doi.org/10.17632/pwctygncrj.4)) and use the five classes (motherboard, battery, RAM, WiFi card, CPU fan). The 100 researcher-collected OOD images.
3. **Open the notebook.** Open from Algorithms to find the latest version or Logs and Project Notebooks for other versions in Colab.
4. **Select the GPU.** *Runtime → Change runtime type → T4 GPU*.
5. **Mount Drive and set paths.** Run the first cells to mount Google Drive and point to the dataset and OOD folders on your local.
6. **Run all cells in order** (*Runtime → Run all*). The notebook covers the full pipeline: blur filtering, preprocessing, partitioning, training the four configurations across three seeds, the embedding audit, OOD evaluation, and the efficiency benchmark.
7. **Trained checkpoints** are available if you only want to run inference.

**Dataset credit.** Hasan, Imran; Rupin, Rezmah Jemima; Fahim, Sajid Faysal; Chowdhury, Mehrab; Rashid, Mohammad Rifat Ahmmad; Ali, Sawkat (2023), “Laptop Components: Image Dataset to Classify Different Components”, Mendeley Data, V2, doi: 10.17632/pwctygncrj.2 (https://doi.org/10.17632/pwctygncrj.4). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The images were filtered, cropped, resized, and split for this project, and the creators have not endorsed it. Code is released under the MIT license (see `LICENSE`).
