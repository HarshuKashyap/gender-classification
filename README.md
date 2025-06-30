# 🎧 Gender Classification Using Acoustic Features

This project uses machine learning to classify gender (male/female) based on acoustic features extracted from voice samples.

## 📁 Dataset

- Dataset used: `voice.csv`
- Each row represents an audio sample with various acoustic features (mean frequency, modulation, pitch, etc.)
- Target label: `male` or `female`

## 🧠 Algorithm Used

- **Random Forest Classifier**
- 80-20 Train/Test split
- Accuracy and classification report are used for evaluation

## 🔧 Technologies

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Seaborn & Matplotlib


## 📊 Output

- Model is trained and evaluated
- Final output includes accuracy score and classification report

pip install -r requirements.txt
py gender_classifier.py



