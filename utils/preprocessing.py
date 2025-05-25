import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


VALID_VARIANTS = {"raw", "stop", "stop_lemma", "stop_stem"}

def clean_text(text, variant="raw"):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{variant}'. Must be one of {VALID_VARIANTS}.")

    text = re.sub(r'[^\w\s@\$]', '', text)  # keep @ and $ (spam indicators)
    # Convert text to lowercase
    text = text.lower()
    tokens = text.split()

    if variant.startswith("stop"):
        tokens = [word for word in tokens if word not in stop_words]

    if variant == "stop_lemma":
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    elif variant == "stop_stem":
        tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)