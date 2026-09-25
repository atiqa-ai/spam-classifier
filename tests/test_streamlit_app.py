"""End-to-end tests that actually execute the Streamlit app.

Streamlit serves a static bootstrap page, so an HTTP 200 tells you nothing about
whether the script ran. ``AppTest`` runs the real script and gives access to the
rendered widget tree and output, which is what these assertions inspect.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(REPO_ROOT / "app.py")


@pytest.fixture(scope="module")
def loaded_app():
    """The app with the model already loaded, before any interaction."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, f"app raised on load: {at.exception}"
    return at


class TestAppRuns:
    def test_runs_without_exception(self, loaded_app):
        assert not loaded_app.exception

    def test_renders_title(self, loaded_app):
        assert any("Spam Classifier" in t.value for t in loaded_app.title)

    def test_has_input_and_button(self, loaded_app):
        assert len(loaded_app.text_area) == 1
        assert len(loaded_app.button) >= 1

    def test_model_artifacts_available(self, loaded_app):
        """A failure to unpickle the pickles must surface, not hide."""
        import app as app_module

        vectorizer, model = app_module.load_artifacts(REPO_ROOT)
        assert vectorizer.__class__.__name__ == "TfidfVectorizer"
        assert model.__class__.__name__ == "MultinomialNB"


class TestPrediction:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Congratulations! You won 1000 free gift cards. Claim now", "Spam"),
            ("FREE entry to win a brand new car! Text WIN to 80086", "Spam"),
            ("Hey are we still meeting for lunch tomorrow?", "Not spam"),
            ("Can you pick up milk on your way home?", "Not spam"),
        ],
    )
    def test_classifies_message(self, loaded_app, message, expected):
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        at.text_area[0].set_value(message)
        at.button[0].click().run()

        assert not at.exception, f"app raised: {at.exception}"
        headers = [h.value for h in at.header] + [h.value for h in at.error] + \
                  [s.value for s in at.success]
        assert expected in headers, f"expected {expected!r} in {headers}"

    def test_spam_uses_error_banner(self, loaded_app):
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        at.text_area[0].set_value("URGENT claim your free prize now")
        at.button[0].click().run()
        assert len(at.error) == 1
        assert at.error[0].value == "Spam"


class TestEmptyInputGuard:
    @pytest.mark.parametrize("blank", ["", "   ", "\n\n"])

    def test_blank_input_warns_instead_of_crashing(self, loaded_app, blank):
        """The original app fed an empty string straight into the vectoriser."""
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        at.text_area[0].set_value(blank)
        at.button[0].click().run()

        assert not at.exception, f"app raised on blank input: {at.exception}"
        assert len(at.warning) == 1
        assert "enter a message" in at.warning[0].value.lower()
        assert not at.error and not at.success, "produced a verdict on empty input"
