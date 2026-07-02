from setuptools import setup, find_packages

setup(
    name="lexisem",
    version="1.0.0",
    description="Lexico-semantic features for sentiment analysis (reference implementation)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "nltk>=3.8", "gensim>=4.3", "scikit-learn>=1.3",
        "numpy>=1.24", "afinn>=0.1",
    ],
    extras_require={"lstm": ["torch>=2.0"]},
    license="MIT",
)
