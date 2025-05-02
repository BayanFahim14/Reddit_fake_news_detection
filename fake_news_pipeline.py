import pandas as pd
import numpy as np
import string
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        text = text.lower()
        text = self.punctuation_pattern.sub("", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and t.isalpha()]
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized)

train_df = pd.read_csv("data/xy_train.csv")
test_df = pd.read_csv("data/x_test.csv")

X = train_df['text']
y = train_df['label']
X_test = test_df['text']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('preprocess', TextPreprocessor()),
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:,1]

print("Train Columns:", train_df.columns.tolist())
print("Test Columns:", test_df.columns.tolist())
# print("ROC-AUC:", roc_auc_score(y_val, y_proba))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))

test_proba = pipeline.predict_proba(X_test)[:,1]
output_df = pd.DataFrame({
    'id': test_df['ID'],
    'label': test_proba
})
output_df.to_csv("outputs/final_predictions.csv", index=False)

joblib.dump(pipeline, "models/fake_news_model.pkl")
