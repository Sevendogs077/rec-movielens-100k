# rec-movielens

![License](https://img.shields.io/badge/License-MIT-blue)

---

### Introduction

A simple implementation of Recommender System based on the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1M/).

---

### Model Zoo

![MF](https://img.shields.io/badge/MF-Done-success)
![GMF](https://img.shields.io/badge/GMF-Done-success)
![NCF](https://img.shields.io/badge/NCF-Building-yellow)
![DeepFM](https://img.shields.io/badge/DeepFM-Planned-lightgrey)
![DIN](https://img.shields.io/badge/DIN-Planned-lightgrey)
![SASRec](https://img.shields.io/badge/SASRec-Planned-lightgrey)
![LightGCN](https://img.shields.io/badge/LightGCN-Planned-lightgrey)
![LLM4Rec](https://img.shields.io/badge/LLM4Rec-Planned-lightgrey)

---

### Requirements

- `Python` >= 3.10
- `PyTorch` >= 2.0.0 (CUDA recommended)
- Other dependencies: `pandas`, `numpy`, `tqdm`, `matplotlib`

---

### Quick Start

#### Training

```bash
python train.py
```

#### Evaluation

```bash
python evaluate.py
```

---

### Project Structure

```text
.        
├── data/                  # Data directory
├── output/                # saved models and figures
├── src/                   # Source code package
│   ├── __init__.py        # Package initialization
│   ├── dataset.py         # Dataset loading logic
│   ├── model.py           # Model architecture definition
│   └── utils.py           # Argument parsing & utility functions
├── .gitignore             # Git ignore rules
├── evaluate.py            # Main evaluation script
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── train.py               # Main training script
```
