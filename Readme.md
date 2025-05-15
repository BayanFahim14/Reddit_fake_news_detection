Here's a concise and accurate `README.md` file specifically tailored to **your current version of the code**, which takes user input and uses a pre-trained model to classify a Reddit post title as fake or genuine:

---

# Reddit Fake News Classifier

This is a simple Python script that predicts whether a Reddit post title is **FAKE** or **GENUINE** using a pre-trained machine learning model.

## How It Works

* You enter a Reddit post title in the terminal.
* The script uses a trained model (`fake_news_model.pkl`) to compute:

  * The probability that the title is fake.
  * A final prediction: **FAKE** or **GENUINE**.

## Prerequisites

* Python 3.6 or later
* Dependencies:

  * `pandas`
  * `joblib`
  * A custom `TextPreprocessor` class in `preprocessing.py`
  * The trained model file (`fake_news_model.pkl`)

Install dependencies using:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` might look like this:

```
pandas
joblib
scikit-learn
```

## File Structure

```
.
├── models/
│   └── fake_news_model.pkl         # Trained ML model (e.g., sklearn Pipeline)
├── preprocessing.py                # Custom text preprocessor (must define TextPreprocessor)
├── predict.py                      # Main prediction script
├── requirements.txt
└── README.md
```

## Usage

1. Run the script:

```bash
python predict.py
```

2. Enter a Reddit post title when prompted:

```
Enter Reddit post title: Scientists confirm Earth is flat
```

3. Example output:

```
Probability of being FAKE: 0.8732
Prediction: FAKE
```

## Important Notes

* The model must be trained on data with the same preprocessing used in this script. If you're using a separate `TextPreprocessor` class, ensure it's applied to the input before prediction.
* If your model is a complete `Pipeline` (with preprocessing and classifier), you don’t need to manually call a preprocessor.

## License

This project is open-source and available under the MIT License.

---

Let me know if you want this adapted for use in a web app, notebook, or API format!
