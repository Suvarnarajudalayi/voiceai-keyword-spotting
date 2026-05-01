# VoiceAI — Keyword Spotting for Low-Power Edge Devices

Deep learning project for the MSc Data Science **Deep Learning Applications** module (CMP-L016) at the University of Roehampton.

A 1D Convolutional Neural Network trained on the Google Speech Commands v0.02 dataset for recognising 35 spoken keywords, with a Flask web demo deployed via ngrok.

---

## Project Overview

- **Task:** Multi-class speech command classification (35 classes)
- **Dataset:** Google Speech Commands v0.02 (~106,000 1-second audio clips)
- **Features:** 40 MFCC coefficients (mean-pooled across time)
- **Model:** 1D CNN with 2 Conv1D + MaxPooling blocks, ~193K parameters (~755 KB)
- **Test accuracy:** 36.75%
- **Deployment:** Flask + ngrok web app with browser-based recording

---

## Repository Contents

```
.
├── SUVARNA_RAJU_DeepLearning_A00073307.ipynb   # Main Colab notebook
├── README.md                                    # This file
└── saved_model/                                 # Generated after running notebook
    ├── keyword_spotting_model.h5
    ├── label_encoder.pkl
    └── scaler.pkl
```

---

## How to Run

### Option 1 — Google Colab (recommended)

1. Open the notebook in Google Colab
2. Set runtime to GPU: `Runtime → Change runtime type → T4 GPU`
3. Upload your `kaggle.json` API token when prompted
4. Set your ngrok auth token in the "Setup ngrok" cell
5. Click `Runtime → Run all`
6. Wait ~30-40 minutes for the full pipeline to complete
7. Click the printed ngrok URL to launch the demo

### Option 2 — Local Python environment

Requirements:
- Python 3.10+
- TensorFlow 2.x, librosa, scikit-learn, Flask, pyngrok, kaggle

```bash
pip install tensorflow librosa scikit-learn flask pyngrok kaggle pandas matplotlib seaborn
```

Then run the notebook cells in order.

---

## Pipeline

1. **Data acquisition** — Kaggle API downloads the Speech Commands v0.02 dataset
2. **Preprocessing** — librosa loads each .wav file at 16 kHz, extracts 40 MFCCs, takes the mean across time → (40,) feature vector
3. **Splitting** — 80/20 train/test split, StandardScaler fitted on training set
4. **Encoding** — LabelEncoder converts 35 string labels to integer indices
5. **Model** — Sequential 1D CNN: Conv1D(64) → MaxPool → Conv1D(128) → MaxPool → Flatten → Dense(128) → Dense(35, softmax)
6. **Training** — Two experiments at 30 epochs (Exp 1: default lr; Exp 2: lr=0.0001)
7. **Evaluation** — Accuracy, precision, recall, F1, per-class report, confusion matrix
8. **Saving** — Model (.h5), LabelEncoder (.pkl), StandardScaler (.pkl)
9. **Deployment** — Flask app with audio upload + browser MediaRecorder, served via ngrok

---

## Results

| Metric | Experiment 1 | Experiment 2 |
|---|---|---|
| Final val accuracy | 36.75% | 34.15% |
| Final val loss | 2.1793 | 2.2995 |
| Epochs | 30 | 30 |
| Learning rate | Adam default (0.001) | 0.0001 |

Best-performing classes (by F1): six (0.62), bird (0.52), eight (0.49)
Worst-performing classes: dog (0.14), down (0.19), no (0.20)

---

## Limitations

- **Mean-pooled MFCC discards temporal information** — the main reason accuracy stays at ~36% rather than the 90%+ achievable with 2D mel-spectrograms
- **No regularisation** (Dropout, BatchNorm) — Experiment 2 shows mild overfitting
- **No data augmentation** — adding time-shift, pitch-perturbation, and noise injection would improve robustness

Future work would address these by switching to a 2D CNN over the full MFCC matrix and adding standard regularisation techniques.

---

## AI Acknowledgement

I used AI tools (Claude by Anthropic) to help debug a Conv1D input shape error in my deployment code and to check my report for internal consistency. All experiments, training, results, and writing are my own work.

---

## Author

Suvarna Raju Dalayi
MSc Data Science — University of Roehampton
Module: CMP-L016 Deep Learning Applications
