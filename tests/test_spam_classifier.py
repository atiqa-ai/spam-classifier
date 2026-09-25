"""Tests for the spam classifier preprocessing and saved artefacts."""

import pickle
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from preprocessing import STOPWORDS, ensure_nltk_data, transform_text  # noqa: E402

ensure_nltk_data()


@pytest.fixture(scope="module")
def artifacts():
    """Load the committed pickles once for the whole module."""
    with open(REPO_ROOT / "vectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)
    with open(REPO_ROOT / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    return vectorizer, model


class TestTransformText:
    def test_lowercases(self):
        assert transform_text("HELLO") == "hello"

    def test_strips_punctuation(self):
        assert "!" not in transform_text("free!!!")

    def test_removes_stopwords(self):
        result = transform_text("the and of a an to in is you it")
        assert result == ""

    def test_stems_words(self):
        # Porter collapses inflections onto a shared stem: caring/care/cares all
        # become "care". (Note it leaves "freed" alone - the -ed step needs a
        # stem of sufficient measure, so "freed" is its own stem.)
        assert transform_text("caring") == "care"
        assert transform_text("cares") == "care"
        assert transform_text("running") == "run"
        assert transform_text("ponies") == transform_text("pony") == "poni"
        assert transform_text("happiness") == transform_text("happy") == "happi"

    def test_keeps_alphanumeric_tokens(self):
        assert "abc123" in transform_text("abc123")

    def test_empty_input(self):
        assert transform_text("") == ""

    def test_whitespace_only_input(self):
        assert transform_text("   \t  ") == ""

    def test_punctuation_only_input(self):
        assert transform_text("!@#$%^&*()") == ""

    def test_output_is_single_spaced(self):
        assert "  " not in transform_text("free   entry   now")

    def test_is_deterministic(self):
        msg = "Congratulations! You won 1000 free gift cards."
        assert transform_text(msg) == transform_text(msg)

    def test_every_output_token_is_stemmed_and_alnum(self):
        tokens = transform_text("Free entry to win a brand NEW car 12345!").split()
        assert tokens
        assert all(t.isalnum() for t in tokens)
        assert not any(t in STOPWORDS for t in tokens)


class TestSavedArtefacts:
    """The committed pickles must load and predict, or the app is broken."""

    def test_vectoriser_is_fitted_tfidf(self, artifacts):
        vectorizer, _ = artifacts
        assert vectorizer.__class__.__name__ == "TfidfVectorizer"
        assert hasattr(vectorizer, "idf_"), "vectoriser is not fitted"

    def test_model_is_multinomial_nb(self, artifacts):
        _, model = artifacts
        assert model.__class__.__name__ == "MultinomialNB"

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Congratulations! You won 1000 free gift cards. Claim now", 1),
            ("FREE entry to win a brand new car! Text WIN to 80086", 1),
            ("Hey are we still meeting for lunch tomorrow?", 0),
            ("Can you pick up milk on your way home?", 0),
            ("See you at the office in an hour", 0),
        ],
    )
    def test_predictions(self, artifacts, message, expected):
        vectorizer, model = artifacts
        got = int(model.predict(vectorizer.transform([transform_text(message)]))[0])
        assert got == expected

    def test_end_to_end_pipeline_shape(self, artifacts):
        vectorizer, model = artifacts
        features = vectorizer.transform([transform_text("free entry")])
        assert features.shape[0] == 1
        assert features.shape[1] == vectorizer.max_features
        assert len(model.predict(features)) == 1


class TestAppModule:
    def test_app_imports_cleanly(self):
        import app  # noqa: F401

    def test_load_artifacts_returns_pair(self):
        import app

        vectorizer, model = app.load_artifacts(REPO_ROOT)
        assert vectorizer is not None
        assert model is not None
