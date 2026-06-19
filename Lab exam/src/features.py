import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow_vectorizer(max_features=5000):
    """Create a Bag-of-Words CountVectorizer."""
    return CountVectorizer(max_features=max_features)


def build_tfidf_vectorizer(max_features=5000):
    """Create a TF-IDF TfidfVectorizer with sublinear scaling."""
    return TfidfVectorizer(max_features=max_features, sublinear_tf=True)


def fit_transform_vectorizer(vectorizer, X_train, X_test):
    """Fit vectorizer on train data and transform both train and test."""
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def get_top_terms_by_class(vectorizer, model, class_index, n=20):
    """Return top N terms most indicative of a given class."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(model, 'coef_'):
        coefs = model.coef_[class_index]
    elif hasattr(model, 'feature_log_prob_'):
        coefs = model.feature_log_prob_[class_index]
    else:
        raise AttributeError("Model has no coef_ or feature_log_prob_")
    top_indices = np.argsort(coefs)[::-1][:n]
    return feature_names[top_indices].tolist()