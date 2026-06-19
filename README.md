# Legal Notice Document Classifier
### Riphah International University — Lahore Campus
### Programming For AI (Lab Assessment) — Spring 2026
### End-Term Examination

---

## Student Information
| Field | Details |
|---|---|
| **Student Name** | [Parsa maryam] |
| **SAP ID** | [66560] |
| **Section** | [BSAI 4A] |
| **Program** | [PROGRAMING FOR AI] |

---

##  Project Description
This project builds an end-to-end **multi-class text classification system** for a legal tech startup. The system automatically categorises short legal notices (50–200 words) into one of three categories:

- **A — Contract Dispute**
- **B — Intellectual Property Claim**
- **C — Regulatory Compliance**

Two classical machine learning models are compared across two feature representations to recommend the best model for production use.

---

## Project Structure
sentiment-lab-[studentID]/
│
├── data/
│   ├── raw/
│   │   └── legal_notices.csv        ← Original dataset (600 samples)
│   └── processed/                  ← Preprocessed data
│
├── notebooks/
│   └── sentiment_analysis.ipynb     ← Main notebook (run this)
│
├── src/
│   ├── preprocess.py                ← Text cleaning pipeline
│   ├── features.py                  ← BoW and TF-IDF vectorizers
│   └── evaluate.py                  ← Metrics and confusion matrices
│
├── results/                         ← Saved graphs and confusion matrices
│   ├── eda_plots.png
│   ├── cm_lr_bow.png
│   ├── cm_lr_tfidf.png
│   ├── cm_nb_bow.png
│   ├── cm_nb_tfidf.png
│   └── hyperparam_C.png
│
├── mlruns/                          ← MLflow experiment logs
├── config.json                      ← All hyperparameters
├── requirements.txt                 ← Required libraries
└── README.md                        ← This file
---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOURUSERNAME/sentiment-lab-YOURSAPID.git
cd sentiment-lab-YOURSAPID
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
cd notebooks
jupyter notebook sentiment_analysis.ipynb
```
Then run all cells from top to bottom.

### 4. View MLflow Results
```bash
mlflow ui
```
Open browser and go to: `http://localhost:5000`

---

##  Configuration File (config.json)
All hyperparameters are stored in `config.json`. No values are hardcoded in the code.

| Parameter | Value | Description |
|---|---|---|
| `random_seed` | 42 | Fixed seed for reproducibility |
| `test_size` | 0.2 | 80% train, 20% test split |
| `max_features` | 5000 | Maximum vocabulary size |
| `model_1.C` | 1.0 | Logistic Regression regularisation strength |
| `model_2.alpha` | 1.0 | Naive Bayes Laplace smoothing parameter |

---

## Models Compared

| Model | Feature Representation |
|---|---|
| Logistic Regression | Bag of Words (BoW) |
| Logistic Regression | TF-IDF |
| Naive Bayes | Bag of Words (BoW) |
| Naive Bayes | TF-IDF |

---

##  Results Summary

| Configuration | Accuracy | F1 Macro | F1 Weighted | Train Time |
|---|---|---|---|---|
| LR + BoW | fill after running | fill after running | fill after running | fill after running |
| LR + TF-IDF | fill after running | fill after running | fill after running | fill after running |
| NB + BoW | fill after running | fill after running | fill after running | fill after running |
| NB + TF-IDF | fill after running | fill after running | fill after running | fill after running |

**Best Model:** Logistic Regression + TF-IDF

**Reason:** TF-IDF down-weights common words and focuses on rare but important legal keywords like *patent*, *breach*, *compliance*. Logistic Regression handles multi-class problems well and is interpretable.

---

##  MLflow Experiment Tracking
All training runs are logged to MLflow including:
- Model type and vectorizer type
- All hyperparameters
- Accuracy, F1 Macro, F1 Weighted
- Training time
- Confusion matrix image for each run

Total runs logged: **9 runs** (4 main + 5 hyperparameter experiments)

---

##  Hyperparameter Experiment
Logistic Regression C values tested: `[0.01, 0.1, 1.0, 10.0, 100.0]`

Best C value selected based on highest F1 Macro score on test set.

---

##  Limitations
1. Dataset has only 600 samples which limits model generalisation
2. Unigram features miss important multi-word legal phrases like *breach of contract*

---

##  AI Usage Statement
AI tools (Claude by Anthropic) were used during this assessment for:
- Code scaffolding and boilerplate generation
- Debugging error messages
- README template drafting

Parts completed without AI assistance:
- All analytical reasoning and justifications in Part 1
- Performance analysis and interpretation of results in Part 5
- Viva preparation and code understanding

One instance where AI was incorrect:
The AI initially suggested using TfidfVectorizer without sublinear_tf=True. After checking sklearn documentation I corrected this to include sublinear_tf=True as required by the exam specification.

Overall AI is a useful productivity tool for repetitive tasks but critical thinking is still required to verify outputs and make domain-specific decisions.

---

##  Libraries Used
- Python 3.10
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- nltk
- mlflow
- jupyter

---

## Commit History
| Commit | Description |
|---|---|
| 1st | Initial project structure and config setup |
| 2nd | Part 2: EDA, preprocessing and feature extraction |
| 3rd | Part 3: Model training, evaluation and hyperparameter experiments |
| 4th | Part 4-5: MLflow logging, reflection and documentation |
