# SMS Spam Classifier

An end-to-end NLP project that trains a machine learning model to tell **spam**
from **ham** (legitimate) SMS messages, built on the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

The notebook walks the full workflow: cleaning → exploratory analysis → text
preprocessing → vectorisation → model comparison → evaluation → saving the model.

## Results

Test-set performance (20% hold-out, `random_state=42`):

| Metric | Value |
| --- | --- |
| Accuracy | **0.982** |
| Precision | **0.963** |

The best single model is **MultinomialNB** on **TF-IDF** features
(`max_features=3000`). The notebook also compares GaussianNB, BernoulliNB,
SVC, K-NN, Decision Tree, Logistic Regression, Random Forest, Bagging, Extra
Trees, Gradient Boosting, XGBoost, and a stacked ensemble.

## Dataset

| | |
| --- | --- |
| Source | UCI SMS Spam Collection |
| Raw messages | 5,572 |
| After removing duplicates | 5,169 |
| Ham (legitimate) | 4,516 |
| Spam | 653 (12.6%) |

The class imbalance (roughly 7:1 toward ham) is exactly what makes this dataset
interesting — a model that predicts "ham" for everything would already reach
87% accuracy, so accuracy alone is not a meaningful signal. Precision on the
spam class is the number that matters here.

## Pipeline

```
spam.csv
   ↓  drop unnamed all-NaN columns, rename v1→output, v2→text
   ↓  drop duplicates
   ↓  exploratory analysis: class balance, length histograms, word clouds
   ↓  transform_text(): lowercase → tokenize → drop punctuation/stopwords → Porter stem
   ↓  TfidfVectorizer(max_features=3000)
   ↓  train/test split (80/20, random_state=42)
   ↓  train 12 classifiers, compare accuracy + precision
   ↓  pick the best, save with pickle
```

The `transform_text` function is the heart of the preprocessing. Stemming matters
more than it might seem here: collapsing "free", "freed", and "freeing" into one
token stops the model from treating them as three unrelated words.

## Requirements

Python 3.9+ and Jupyter. The dataset is already included, so no download is needed.

```bash
git clone https://github.com/atiqa-ai/spam-classifier.git
cd spam-classifier

python -m venv .venv
# Windows:     .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

### NLTK data

The notebook downloads two NLTK corpora on first use. If you prefer to install
them ahead of time:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

The notebook's first run downloads these automatically if they are missing.

## Usage

Open `SMS_Spam_classification.ipynb` and run all cells.

The notebook locates `spam.csv` by walking up from the working directory, so it
runs whether you start Jupyter from the project folder or a subfolder of it.

### Loading the saved model

```python
import pickle

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

text = "Congratulations! You won a free prize, claim now"
prediction = model.predict(tfidf.transform([text]))[0]
print("spam" if prediction == 1 else "ham")
```

Both files are committed, so no training run is needed to use the model.

## Web app

A [Streamlit](https://streamlit.io) front end is included:

```bash
streamlit run app.py
```

Then open http://localhost:8501, paste a message, and press **Predict**.

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit front end |
| `preprocessing.py` | The `transform_text` pipeline, importable and tested |
| `tests/test_spam_classifier.py` | Preprocessing and artefact tests |
| `tests/test_streamlit_app.py` | End-to-end tests that run the real app |

`preprocessing.py` is shared by the app and the tests, so the two cannot drift
apart. The tests confirm it produces byte-identical output to the original
notebook implementation, which matters: a change here silently changes every
prediction.

The app is tested with Streamlit's own `AppTest` harness, which executes the
real script and inspects the rendered output. Serving HTTP 200 is not enough —
that only proves the bootstrap page loaded.

## Docker

```bash
docker build -t spam-classifier .
docker run --rm -p 8501:8501 spam-classifier
```

Then open http://localhost:8501.

## Continuous integration

The GitHub Actions workflow runs the full test suite on every push, so a
regression in the preprocessing or a pickle that stops loading fails the build
rather than reaching a user.

## Known limitations

- **Pickles are version-sensitive.** `model.pkl` and `vectorizer.pkl` were
  generated with the exact versions in `requirements.txt` (scikit-learn 1.9.1).
  Loading them under a very different scikit-learn can fail; re-run the notebook
  to regenerate if that happens.
- **It is a bag-of-words model.** The classifier sees word frequency, not word
  order or intent. It comfortably handles the obvious spam in the dataset, but
  novel phrasings can slip through — a bank-phishing style message
  ("URGENT! Your bank account will be suspended, call 0900...") was classified
  as ham during testing. That is a real limitation of TF-IDF + Naive Bayes, not
  a bug.
- **English SMS only.** The dataset is British mobile spam, so the model has no
  coverage for other languages or for email phishing.
- **The 98.2% accuracy is on a random split of one small dataset.** It is a
  measure of fitting this collection, not of production performance.

## What I learned

- Accuracy is misleading on imbalanced data; precision and the confusion matrix tell the real story.
- Preprocessing does more for text models than hyperparameter tuning.
- Word clouds and length histograms revealed the signal before any model did — spam messages are far longer, and keywords like "free", "claim", and "prize" dominate.
- Naive Bayes is a strong, fast baseline for bag-of-words text classification.

## Project files

| File | Purpose |
| --- | --- |
| `SMS_Spam_classification.ipynb` | The full workflow |
| `spam.csv` | UCI SMS Spam Collection |
| `model.pkl` | Trained `MultinomialNB` classifier |
| `vectorizer.pkl` | Fitted `TfidfVectorizer` |
| `app.py` | Streamlit web app |
| `preprocessing.py` | Shared text preprocessing |
| `tests/` | 33 tests, including end-to-end app tests |


