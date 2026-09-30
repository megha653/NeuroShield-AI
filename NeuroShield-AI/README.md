# NeuroShield AI

An explainable ML/NLP prototype for detecting social-engineering language in text messages.

## What is included
- TF-IDF feature extraction
- Logistic Regression classifier
- Train/test split and evaluation
- Accuracy, precision, recall, F1 and confusion matrix output
- Explainable suspicious-term contributions
- Streamlit dashboard

## Run
```bash
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

## Data
`data/messages.csv` is a small synthetic demonstration dataset included only so the repository runs immediately. For a serious portfolio version, replace it with a properly sourced and documented dataset and retrain/evaluate the model.

## Scope
This ZIP is deliberately the **ML/NLP MVP**, not a fake finished multimodal deepfake system. The image/deep-learning branch (e.g. MobileNetV2 transfer learning) should only be added after a real image dataset is selected and evaluated.

## Interview summary
Input text -> preprocessing/TF-IDF -> numerical features -> Logistic Regression -> probability -> feature-level explanation.

## Disclaimer
Educational prototype. Do not use the output as a definitive security verdict.
