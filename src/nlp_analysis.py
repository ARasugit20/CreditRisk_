"""NLP frequency analysis on borrower description/title text by loan outcome."""

from __future__ import annotations

import logging
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import FreqDist, bigrams, word_tokenize
from nltk.corpus import stopwords

from plot_style import apply_plot_style
from preprocessing import VALID_TARGETS
from utils import DATA_DIR, OUTPUTS_DIR, TARGET_COLUMN, configure_logging, ensure_directories


def ensure_nltk_assets() -> None:
    """Download tokenizer/stopword assets if missing."""

    for resource in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_and_tokenize(series: pd.Series) -> list[str]:
    """Tokenize text into lowercase alphabetic terms without stopwords."""

    english_stopwords = set(stopwords.words("english"))
    tokens: list[str] = []
    for text in series.dropna().astype(str):
        raw = word_tokenize(text.lower())
        tokens.extend(
            token
            for token in raw
            if re.fullmatch(r"[a-z]+", token) and token not in english_stopwords and len(token) > 2
        )
    return tokens


def plot_top_words(
    charged_words: list[str],
    paid_words: list[str],
    text_column: str,
) -> None:
    """Save side-by-side top-20 frequency bar charts by target class."""

    apply_plot_style()
    charged_freq = FreqDist(charged_words).most_common(20)
    paid_freq = FreqDist(paid_words).most_common(20)

    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, title, values in [
        (axes[0], "Charged Off", charged_freq),
        (axes[1], "Fully Paid", paid_freq),
    ]:
        words = [word for word, _ in values][::-1]
        counts = [count for _, count in values][::-1]
        axis.barh(words, counts)
        axis.set_title(f"Top words ({title})")
        axis.set_xlabel("Frequency")
        axis.set_ylabel("Token")
    figure.suptitle(f"Top borrower terms from `{text_column}`")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "nlp_top_words.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_top_bigrams(charged_words: list[str], paid_words: list[str], text_column: str) -> None:
    """Plot top-20 bigrams for each outcome class."""

    apply_plot_style()
    charged_pairs = Counter([" ".join(pair) for pair in bigrams(charged_words)]).most_common(20)
    paid_pairs = Counter([" ".join(pair) for pair in bigrams(paid_words)]).most_common(20)

    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, title, values in [
        (axes[0], "Charged Off", charged_pairs),
        (axes[1], "Fully Paid", paid_pairs),
    ]:
        terms = [term for term, _ in values][::-1]
        counts = [count for _, count in values][::-1]
        axis.barh(terms, counts)
        axis.set_title(f"Top bigrams ({title})")
        axis.set_xlabel("Frequency")
        axis.set_ylabel("Bigram")
    figure.suptitle(f"Top borrower bigrams from `{text_column}`")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "nlp_top_bigrams.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_frequency_tables(charged_words: list[str], paid_words: list[str]) -> None:
    """Persist top unigram and bigram frequencies to CSV outputs."""

    unigram_rows = []
    for label, words in [("charged_off", charged_words), ("fully_paid", paid_words)]:
        for token, count in FreqDist(words).most_common(100):
            unigram_rows.append({"outcome": label, "token": token, "frequency": count})
    pd.DataFrame(unigram_rows).to_csv(OUTPUTS_DIR / "nlp_unigram_frequencies.csv", index=False)

    bigram_rows = []
    for label, words in [("charged_off", charged_words), ("fully_paid", paid_words)]:
        for token, count in Counter([" ".join(pair) for pair in bigrams(words)]).most_common(100):
            bigram_rows.append({"outcome": label, "bigram": token, "frequency": count})
    pd.DataFrame(bigram_rows).to_csv(OUTPUTS_DIR / "nlp_bigram_frequencies.csv", index=False)


def main() -> None:
    """Run NLP analysis on `desc` or fallback `title` text fields."""

    configure_logging()
    ensure_directories()
    ensure_nltk_assets()

    frame = pd.read_csv(DATA_DIR / "lending_club_sample.csv", low_memory=False)
    frame = frame.loc[frame[TARGET_COLUMN].isin(VALID_TARGETS)].copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map(VALID_TARGETS).astype(int)

    text_column = "desc" if "desc" in frame.columns and frame["desc"].notna().any() else "title"
    if text_column not in frame.columns:
        raise ValueError("NLP analysis requires either `desc` or `title` in source data.")

    charged_text = frame.loc[frame[TARGET_COLUMN] == 1, text_column]
    paid_text = frame.loc[frame[TARGET_COLUMN] == 0, text_column]
    charged_words = clean_and_tokenize(charged_text)
    paid_words = clean_and_tokenize(paid_text)

    if not charged_words or not paid_words:
        raise ValueError(f"Insufficient non-empty text tokens in `{text_column}` for NLP analysis.")

    save_frequency_tables(charged_words, paid_words)
    plot_top_words(charged_words, paid_words, text_column=text_column)
    plot_top_bigrams(charged_words, paid_words, text_column=text_column)
    logging.info("Saved NLP outputs in %s using text column `%s`", OUTPUTS_DIR, text_column)


if __name__ == "__main__":
    main()
