import numpy as np
from collections import Counter
from typing import List, Tuple, Dict

class EmbeddingsGenerator:
    def create_word_embeddings(self, preprocessed_sentence: List[List[str]],
                               embedding_dim: int = 100) -> Tuple[int, Dict[str, int], np.ndarray]:
        word_counts = Counter()
        for sentence in preprocessed_sentence:
            word_counts.update(sentence)

        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        word_to_index = {word: index for index, word in enumerate(vocab)}

        vocab_size = len(vocab)
        embeddings_matrix = np.random.rand(vocab_size, embedding_dim)
        return vocab_size, word_to_index, embeddings_matrix

# example usage
#if __name__ == "__main__":
    #from dataset import generate_sentences
    #topic = "Artificial Intelligence"
    #sentences = generate_sentences(topic, num_sentences=20)
    #from preprocess import preprocess_sentences
    #preprocessed_sentences = preprocess_sentences(sentences)

    #vocab_size, word_to_index, embeddings_matrix = create_word_embeddings(preprocessed_sentences)
    #print(f'Vocabulary size: {vocab_size}')
    #print(f'Example word index: {word_to_index["artificial"]}')
    #print(f"Embeddings matrix shape: {embeddings_matrix.shape}")