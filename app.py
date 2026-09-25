"""Streamlit front end for the SMS/email spam classifier.

Run from the project root:

    streamlit run app.py

The model artefacts are located relative to this file, so the app behaves the
same whichever directory Streamlit is launched from.
"""

import pickle
from pathlib import Path

import streamlit as st

from preprocessing import ensure_nltk_data, transform_text

HERE = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts(directory: Path):
    """Load and cache the trained vectoriser and classifier.

    ``st.cache_resource`` keeps this to a single unpickle per server process
    rather than one per rerun.
    """
    with open(directory / "vectorizer.pkl", "rb") as handle:
        vectorizer = pickle.load(handle)
    with open(directory / "model.pkl", "rb") as handle:
        model = pickle.load(handle)
    return vectorizer, model


def main() -> None:
    ensure_nltk_data()
    vectorizer, model = load_artifacts(HERE)

    st.title("SMS / Email Spam Classifier")
    st.caption(
        "Multinomial Naive Bayes over 3,000 TF-IDF features, trained on the "
        "UCI SMS Spam Collection."
    )

    message = st.text_area("Enter the message", height=120)

    if st.button("Predict", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
            return

        prediction = model.predict(vectorizer.transform([transform_text(message)]))[0]
        if prediction == 1:
            st.error("Spam")
        else:
            st.success("Not spam")

    with st.expander("How this works"):
        st.markdown(
            "The message is lower-cased, tokenised, stripped of punctuation and "
            "stopwords, and reduced to word stems. The result is turned into "
            "TF-IDF features and classified.\n\n"
            "This is a bag-of-words model, so it judges vocabulary and length "
            "rather than intent. Novel phrasings can slip through."
        )


if __name__ == "__main__":
    main()
