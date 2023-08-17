# custom_word2vec_oop/tests/test_word2vec.py

import numpy as np
import pytest
from preprocess import Preprocessor
from embeddings import EmbeddingsGenerator
from dataset import DatasetGenerator
from word2vec import SkipgramTrainer, CBOWTrainer

class TestWord2Vec:
    @classmethod
    def setup_class(cls):
        cls.dataset_generator = DatasetGenerator()
        cls.preprocessor = Preprocessor()
        cls.embeddings_generator = EmbeddingsGenerator()
        cls.cbow_trainer = CBOWTrainer()
        cls.skipgram_trainer = SkipgramTrainer()

    def test_train_cbow(self):
        topic = "Artificial Intelligence"
        sentences = self.dataset_generator.generate_sentences(topic, num_sentences=20)
        preprocessed_sentences = self.preprocessor.preprocess_sentences(sentences)
        vocab_size, word_to_index, embeddings_matrix = self.embeddings_generator.create_word_embeddings(preprocessed_sentences)

        original_embeddings = embeddings_matrix.copy()
        updated_embeddings = self.cbow_trainer.train_cbow(preprocessed_sentences, word_to_index, embeddings_matrix)

        assert np.all(original_embeddings != updated_embeddings)

    def test_train_skipgram(self):
        topic = "Artificial Intelligence"
        sentences = self.dataset_generator.generate_sentences(topic, num_sentences=20)
        preprocessed_sentences = self.preprocessor.preprocess_sentences(sentences)
        vocab_size, word_to_index, embeddings_matrix = self.embeddings_generator.create_word_embeddings(preprocessed_sentences)

        original_embeddings = embeddings_matrix.copy()
        updated_embeddings = self.skipgram_trainer.train_skipgram(preprocessed_sentences, word_to_index, embeddings_matrix)

        assert np.all(original_embeddings != updated_embeddings)

    def test_similarity(self):
        topic = "Artificial Intelligence"
        sentences = self.dataset_generator.generate_sentences(topic, num_sentences=20)
        preprocessed_sentences = self.preprocessor.preprocess_sentences(sentences)
        vocab_size, word_to_index, embeddings_matrix = self.embeddings_generator.create_word_embeddings(preprocessed_sentences)
        updated_embeddings_matrix = self.cbow_trainer.train_cbow(preprocessed_sentences, word_to_index, embeddings_matrix)

        target_word = "intelligence"
        target_word_idx = word_to_index[target_word]
        target_embedding = updated_embeddings_matrix[target_word_idx]

        similarity_scores = np.dot(updated_embeddings_matrix, target_embedding)
        similar_indices = np.argsort(similarity_scores)[::-1]

        assert target_word in [word for word, index in word_to_index.items() if index == similar_indices[0]]

if __name__ == "__main__":
    pytest.main()
