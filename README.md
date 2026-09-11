# Overcoming Catastrophic Forgetting in Self-Organizing Maps

This repository contains the code accompanying the paper *"Overcoming Catastrophic
Forgetting in Self-Organizing Maps."* It implements the **Fisher SOM (FSOM)**
algorithm — an Elastic Weight Consolidation (EWC)-based weight consolidation
mechanism for Self-Organizing Maps (SOMs) — along with the two proposed evaluation
metrics, **BMU Weight Shift Error (BWSE)** and **Topography Preservation Error
(TPE)**, and the full suite of experiments used to validate them.

## Repository Structure

```
.
├── algorithms/     # Implementation of FSOM and other relevant algorithms
├── datasets/       # Place raw downloaded datasets here
├── evaluation/     # Helper functions for SOM algorithm evaluation
├── experiments/    # Experiment scripts (Experiments I–VII from the paper)
├── features/       # Feature extraction scripts
├── loaders/        # Load saved features to train SOM networks
├── results/        # Save experiment results (results used in the paper are already saved)
├── util/           # Helper functions
└── README.md
```

## 1. Download the Datasets

Download the following dataset and place it in the `datasets/` folder (create the folder if it's not available):

- **CUB-200-2011** — Caltech-UCSD Birds-200-2011.

Your `datasets/` folder should look like this once it is downloaded:

```
datasets/
└── cub_200_2011/
```

The **CUB-200-2011** images should be in:

```
datasets/
└── cub_200_2011/
    └── images/
```

> **Note:** Refer to each dataset's official page for licensing terms and download
> instructions. Do not commit raw dataset files to the repository.

## 2. Extract Features

Once the datasets are in place, run the appropriate scripts in the `features/`
folder to extract feature embeddings and save them as `.npz` files. This project
uses pretrained ResNet architectures (ResNet-18 for CIFAR-100, ResNet-50 for
CUB-200-2011) to extract embeddings, matching the setup described in the paper.

```bash
python features/<dataset_name>/feature_extractor.py
```

Extracted `.npz` files should be saved so that the experiment scripts in
`experiments/` can load them directly — check each extractor script's output
path before running experiments.

## 3. Run Experiments

Each experiment from the paper (Experiments I–VII) is implemented as a Python
`unittest` module inside the `experiments/` folder. Run an experiment with:

```bash
python -m unittest <file path of the experiment>
```

For example:

```bash
python -m unittest experiments\\experiment_1\\experiment_1.py
python -m unittest experiments\\experiment_2\\experiment_2_C.py
```

Each experiment will generate the relevant plots/metrics (BWSE, TPE, AIA, and the
Combined Metric) as described in the paper.

## Requirements

- Python 3.9+
- PyTorch (with torchvision, for the pretrained ResNet feature extractors)
- NumPy
- Matplotlib (for reproducing the figures)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Feature extraction must be run **before** any experiment, since the experiment
  scripts expect precomputed `.npz` feature files as input rather than raw images.
- Random seeds are fixed within each experiment script for reproducibility; see
  individual experiment files for the exact seed values used.
- This is research code intended to reproduce the paper's results and is not
  optimized for production use.
