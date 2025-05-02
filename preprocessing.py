import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

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