import re
import numpy as np

def create_unique_word_dict(text: list) -> dict:
    """
    A method that creates a dict where keys are unique words and values are indices
    """
    # get all the unique words from our text and sort them alphabetically
    words = list(set(text))
    words.sort()
    # create a dict for unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({word: i})
    return unique_word_dict

#text = ["Please", "replace", "any", "placeholder", "variable", "names", "and","add", "the", "necessary", "import", "statements",  "for", "the", "functions", "and"]

def text_preprocess(text:list,
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_“~''',
    stop_words=['and', 'a', 'is', 'the', 'in', 'be', 'will']
    )->list:
    for x in text:
        if x in punctuations:
            text = text.replace(x, "")
    # remove words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)
    # remove digits
    text = re.sub(r'[0-9]+', '', text)
    # clean the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    # convert all our text to a list
    text = text.split(' ')
    # drop stop words
    text = [x for x in text if x not in stop_words]
    return text

# functions to find the most similar word
def euclidian(vec1: np.array, vec2: np.array) -> float:
    """
    calculate the euclidian distance between 2 vectors
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def find_similar(word: str, embedding_dict: dict, top_n=10) -> list:
    """
    find the most similar word based on the learnt embeddings
    """
    dist_dict = {}
    word_vector = embedding_dict.get(word, [])
    if len(word_vector) > 0:
        for key, value in embedding_dict.items():
            if key != word:
                dist = euclidian(word_vector, value)
                dist_dict.update({key: dist})
    return sorted(dist_dict.items(), key=lambda x: x[1])[0:top_n]
