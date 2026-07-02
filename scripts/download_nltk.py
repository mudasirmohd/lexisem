"""Download the NLTK resources this project needs. Run once after install."""
import nltk

RESOURCES = [
    "movie_reviews", "twitter_samples", "opinion_lexicon", "wordnet", "omw-1.4",
    "punkt", "punkt_tab", "stopwords",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    # only needed for the local-training embedding fallback:
    "brown", "reuters", "gutenberg", "webtext",
]

if __name__ == "__main__":
    for r in RESOURCES:
        ok = nltk.download(r, quiet=True)
        print(("OK  " if ok else "FAIL"), r)
