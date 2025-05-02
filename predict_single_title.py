import joblib
import pandas as pd
from preprocessing import TextPreprocessor

model = joblib.load("models/fake_news_model.pkl")

title = input("Enter Reddit post title: ")
title = pd.Series([title])

proba = model.predict_proba(title)[0][1]
label = model.predict(title)[0]

print(f"\nProbability of being FAKE: {proba:.4f}")
print("Prediction:", "FAKE" if label == 1 else "GENUINE")
