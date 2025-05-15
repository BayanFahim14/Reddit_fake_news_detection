Here’s a tailored `README.md` for your full Reddit fake news detection project, including the `TextPreprocessor` you've defined in `preprocessing.py`.

---

# Reddit Fake News Detector

This project is a command-line tool that predicts whether a Reddit post title is **FAKE** or **GENUINE** using a machine learning model trained on preprocessed text data.

## Features

* Accepts user input (Reddit post titles) via terminal
* Applies custom text preprocessing (tokenization, stopword removal, lemmatization)
* Uses a trained ML model to output:

  * Probability of the post being fake
  * Binary classification: FAKE or GENUINE

## Project Structure

```
.
├── models/
│   └── fake_news_model.pkl         # Pre-trained model (can be a pipeline or standalone)
├── preprocessing.py                # TextPreprocessor class for cleaning input
├── predict.py                      # CLI tool for prediction
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/reddit-fake-news-detector.git
   cd reddit-fake-news-detector
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   pandas
   joblib
   nltk
   scikit-learn
   ```

4. **Download necessary NLTK data:**

   The script automatically downloads NLTK resources on first run:

   ```python
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

Run the prediction script:

```bash
python predict.py
```

When prompted, enter a Reddit post title:

```
Enter Reddit post title: Scientists confirm Earth is flat
```

Example output:

```
Probability of being FAKE: 0.8614
Prediction: FAKE
```

## Custom Text Preprocessing

The preprocessing is handled by `TextPreprocessor` in `preprocessing.py`. It performs:

* Lowercasing
* Punctuation removal
* Tokenization
* Stopword filtering
* Lemmatization

This preprocessing ensures that input is cleaned and formatted the same way it was during training.

## Notes

* The model (`fake_news_model.pkl`) should be trained on text preprocessed in the same way using `TextPreprocessor`.
* If your model is saved as a complete `Pipeline`, it will apply preprocessing automatically, and you can skip manual transformation.

## License

This project is licensed under the MIT License.

---

Let me know if you'd like to include instructions for training the model or converting this into a web app.
