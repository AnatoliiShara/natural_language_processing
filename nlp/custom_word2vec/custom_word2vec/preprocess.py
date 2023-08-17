import spacy
from typing import List

class Preprocessor:
    def __init__(self):
        self.nlp = nlp = spacy.load("en_core_web_sm")
    def preprocess_sentences(self,sentences: List[str]) -> List[List[str]]:
        preprocessed_sentences = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            preprocessed_sentence = [token.lemma_ for token in doc if not token.is_stop]
            preprocessed_sentences.append((preprocessed_sentence))
        return preprocessed_sentences


# Example usage
#if __name__ == "__main__":
    #from dataset import generate_sentences
    #topic = "Artificial Intelligence"
    #sentences = generate_sentences(topic, num_sentences=20)

    #preprocessed_sentences = preprocess_sentences(sentences)
    #for i, preprocessed_sentence in enumerate(preprocessed_sentences, start=1):
        #print(f"{i}. {' '.join(preprocessed_sentence)}")

