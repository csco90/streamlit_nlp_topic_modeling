import pandas as pd
import numpy as np
import re


def clean_doc(doc):
    """function to clean, lemmetize and remove stop words
    input parameters:
    doc: text corpora"""

    import nltk
    from nltk.tokenize import RegexpTokenizer
    import unicodedata
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
    from sklearn.cluster import KMeans
    import pickle

    stopwords = nltk.corpus.stopwords.words('english')

    nltk.download('wordnet')

    doc = doc.lower()
    doc = re.sub(r"(\n+)|(\s+)", " ", doc)
    doc = re.sub(r"asp.net", "aspnet", doc)
    doc = re.sub(r"c#", "ccharp", doc)
    doc = re.sub(r"c\\+\\+", "cplus", doc)
    doc = re.sub(r"[\W_]+", " ", doc)
    doc = doc.strip()

    tokenizer = RegexpTokenizer(r"\w+", gaps=False)
    doc = tokenizer.tokenize(doc)

    doc = [word for word in doc if word not in stopwords]

    wln = WordNetLemmatizer()
    doc = [wln.lemmatize(word) for word in doc]

    doc = " ".join(doc)

    return doc

def get_topic(prediction):
    topics = [
        [""],

        ['network', 'system', 'application', 'learn', 'security'],

        ['java', 'programming', 'language', 'ccharp', 'code'],

        ['sql', 'database', 'oracle', 'sql server', 'server'],
        ['data', 'analysis', 'machine learning', 'hadoop', 'learning'],

        ['android', 'apps', 'io', 'app', 'iphone'],

        ['web', 'application', 'javascript', 'web application', 'php'],

        ['window', 'linux', 'system', 'mac', 'excel'],

        ['exam', 'study', 'certification', 'study guide', 'test'],

        ['python', 'pi', 'raspberry', 'raspberry pi', 'programming'],

        ['game', 'game development', 'unity', 'development', '2d']]

    if (prediction == 0):
        topic = topics[1]
    if (prediction == 1):
        topic = topics[2]
    if (prediction == 2):
        topic = topics[3]
    if (prediction == 3):
        topic = topics[4]
    if (prediction == 4):
        topic = topics[5]
    if (prediction == 5):
        topic = topics[6]
    if (prediction == 6):
        topic = topics[7]
    if (prediction == 7):
        topic = topics[8]
    if (prediction == 8):
        topic = topics[9]
    if prediction == 9:
        topic = topics[10]
    return " ".join(topic)