<div align="center">

# Convolutional Neural Networks for E-Waste Laptop Component Classification
### DenseNet-121 vs MobileNet-V2 with Focal Loss and Progressive Unfreezing

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)
![Dataset License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-2E8B57?style=for-the-badge)
![Code License](https://img.shields.io/badge/Code-MIT-blue?style=for-the-badge)

</div>

---

> A thesis project by Kirk Adrian E. Sunico, Prince Jeffrey Villamil, Radzie R. Bagallon, and John Paul Q. Tomas (adviser)
> Mapúa University, School of Information Technology, Makati, Philippines, 2026

## Overview

This project evaluates convolutional neural networks for classifying five laptop components found in e-waste: **motherboard, battery, RAM, WiFi card, and CPU fan**. It compares **DenseNet-121** with the lightweight **MobileNet-V2** and tests whether **focal loss** and **progressive unfreezing** improve performance.

Four configurations were trained with three random seeds each, then evaluated on:

- an in-distribution test set (65 images),
- an embedding-similarity audit for near-duplicate images across data splits,
- a 100-image out-of-distribution (OOD) set of uncropped, researcher-collected images (50 white background, 50 messy background), and
- an efficiency benchmark (model size, inference latency, FLOPs).

**Data:** 700 images in 5 classes, reduced to 643 after a Laplacian blur filter, split 514 / 64 / 65 into training / validation / test.

---

## Results

| Finding | Result |
|---|---|
| In-distribution test accuracy | 100% for all configurations (ceiling effect) |
| Embedding-similarity audit | Near-duplicate images crossed the data splits, so the test set could not separate the models |
| OOD accuracy, overall | 67.0% to 71.0%, with no significant difference between configurations after Holm correction |
| Efficiency | MobileNet-V2 was about 2.8x smaller, 2.9 to 3.2x faster, and used about 9.3x fewer FLOPs than the DenseNet-121 variants |

**OOD accuracy by background**

| Configuration | White background | Messy background |
|---|---|---|
| MobileNet-V2 baseline | 75.3% | 64.0% |
| DenseNet-121 baseline | 76.0% | 62.7% |
| Frozen control | 78.7% | 55.3% |
| Unfreeze | 78.0% | 64.0% |

**Takeaway:** MobileNet-V2 is the most practical choice for controlled deployment, but none of the evaluated models is yet reliable on cluttered backgrounds.

**Limitations:** the dataset is small, near-duplicates inflated the in-distribution scores, and training runs were not seeded, so exact numbers may vary slightly on re-run.

---

## Setup (Google Colab)

The notebooks are designed to run in Google Colab, so no local installation is needed.

1. **Get the files.** Clone this repository or download it as a ZIP, then upload the folder to your Google Drive.
2. **Get the dataset.** Download the dataset from [TODO: cloud drive link] and place it in `Dataset/` on your Drive.
3. **Open a notebook.** In Colab, open a notebook from `Logs and Project Notebooks/` [TODO: or use the badge: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TODO-notebook-link)`].
4. **Select a GPU.** Go to *Runtime → Change runtime type* and choose a GPU.
5. **Mount Drive and set the path.** Run the first cells to mount Google Drive and point to your dataset folder [TODO: name the variable or cell].
6. **Install dependencies** if the notebook asks, then run all cells in order: *Runtime → Run all*.

**Dataset credit:** the original images are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), credited to [TODO: dataset authors and Mendeley Data link]. The images were filtered, cropped, resized, and split for this project. The creators have not endorsed it. Code is released under the MIT license (see `LICENSE`).
