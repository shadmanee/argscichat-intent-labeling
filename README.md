# Structured Probabilistic Modeling for Intent Labeling in Argumentative Scientific Dialogues
Comparing Naive Bayes, HMM, MEMM, and Linear-chain CRF for sentence-level intent
labeling in ArgSciChat, a dataset of argumentative scientific dialogues about NLP papers.
The goal is to study whether modelling intent transitions across dialogue turns improves
over a flat, non-sequential baseline.
 
---
 
## Repository Structure
 
```
root/
├── data/
│   ├── raw/                          # Downloaded dataset (see Setup)
│   └── processed/                    # Intermediate and final processed files
├── models/
│   └── dialogue_hmm.py               # Custom HMM implementation
└── notebooks/
    ├── preprocessing/
    │   ├── data_exploration.ipynb         # EDA, dialogue ID engineering, label filtering
    │   └── cleaning.ipynb                 # Text normalization, sorting, sanity checks
    └── training/
        ├── hmm_training_1.ipynb
        ├── memm_training_1.ipynb
        ├── crf_training_1.ipynb
        └── nb_training_1.ipynb
```
 
---
 
## Dataset
 
ArgSciChat: https://github.com/federicoruggeri/argscichat_project
 
41 dialogues, 750 sentences (after filtering), 5 intent labels:
`reply_info`, `ask_info`, `give_opinion`, `ask_rebuttal`, `ask_suggestion`

- Note: `intro` and `outro` labels were discarded
 
---
 
## Requirements
 
```bash
pip install -r requirements.txt
pip install -e .
```
 
---
 
## Setup and Run
 
### 1. Download the dataset
 
Run `dataset/download_dataset.py` once before anything else.
 
### 2. Run preprocessing notebooks in order
 
1. notebooks/preprocessing/data_exploration.ipynb   → produces data/processed/processed_df_0.csv
2. notebooks/preprocessing/cleaning.ipynb           → produces data/processed/processed_df_1.csv
 
### 3. Run training notebooks (any order)
 
- notebooks/training/nb_training_1.ipynb
- notebooks/training/hmm_training_1.ipynb
- notebooks/training/memm_training_1.ipynb
- notebooks/training/crf_training_1.ipynb
 
All training notebooks read from data/processed/processed_df_1.csv and are
self-contained — feature extraction is defined inside each notebook.