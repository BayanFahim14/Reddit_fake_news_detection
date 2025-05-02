# Reddit_fake_news_detection
This project focuses on detecting fake news in Reddit posts using machine learning and natural language processing (NLP). It aims to identify misleading or false news articles shared on Reddit by classifying posts as real or fake.

# 📰 Fake News Detection on Reddit Posts

This project focuses on detecting fake news based on the **title** of Reddit posts using a machine learning pipeline that includes data preprocessing, TF-IDF vectorization, and an XGBoost classifier.

---

## 📁 Project Structure

```
.
├── data/
│   ├── xy_train.csv         # Training data with 'text' and 'label'
│   └── x_test.csv           # Test data with 'text' and 'ID'
├── models/
│   └── fake_news_model.pkl  # Trained pipeline saved as a pickle
├── outputs/
│   └── final_predictions.csv # Output probabilities for test set
├── predict_single_title.py  # Script to predict one Reddit post title
├── fake_news_pipeline.py    # Main training and preprocessing pipeline
├── preprocessing.py         # (Assumed) Contains `TextPreprocessor` class
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure the following libraries are installed:

```bash
pip install pandas numpy scikit-learn xgboost nltk joblib
```

Also, download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🏗️ Model Pipeline (`fake_news_pipeline.py`)

1. **Text Preprocessing**:
   - Lowercasing, punctuation removal
   - Tokenization, stopword removal, and lemmatization

2. **Vectorization**:
   - TF-IDF with n-gram range (1,2) and max 5000 features

3. **Classification**:
   - XGBoost Classifier

4. **Output**:
   - Trains on labeled data and predicts probabilities on test data
   - Saves model as `models/fake_news_model.pkl`
   - Writes predictions to `outputs/final_predictions.csv`

---

## 🔍 Predicting Single Titles (`predict_single_title.py`)

Allows real-time prediction for a single Reddit post title:

```bash
python predict_single_title.py
```

**Output:**
- Probability of being fake
- Final prediction (FAKE or GENUINE)

---

## 📊 Example Output

```
Enter Reddit post title: The government bans free speech again
Probability of being FAKE: 0.8723
Prediction: FAKE
```

---

## 📌 Notes

- The `TextPreprocessor` class is embedded within the pipeline and ensures the same processing during training and inference.
- The model can be extended to consider additional features like post content or metadata.

---

## 🧠 Model Performance

Commented out lines in `fake_news_pipeline.py` can be uncommented to print ROC-AUC, classification report, and confusion matrix for validation data.

---

## 📜 License

This project is for academic purposes only. All rights reserved by the author(s).

