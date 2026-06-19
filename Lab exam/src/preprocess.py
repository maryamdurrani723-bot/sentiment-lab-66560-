import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def preprocess_text(text):
    """Clean and preprocess a single text document."""
    
    # Step 1 - Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Step 2 - Lowercase
    text = text.lower()
    
    # Step 3 - Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Step 4 - Tokenise
    tokens = text.split()
    
    # Step 5 - Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Step 6 - Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)


def preprocess_corpus(texts):
    """Apply preprocess_text to a list of documents."""
    return [preprocess_text(doc) for doc in texts]