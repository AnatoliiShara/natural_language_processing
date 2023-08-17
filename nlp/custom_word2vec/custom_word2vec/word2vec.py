import numpy as np
from typing import List, Dict
from dataset import DatasetGenerator
from preprocess import Preprocessor
from embeddings import EmbeddingsGenerator

class CBOWTrainer:
    def train_cbow(self, preprocessed_sentences: List[List[str]],
                   word_to_index: Dict[str, int],
                   embeddings_matrix: np.ndarray, context_size: int = 2) -> np.ndarray:
        vocab_size, embedding_dim = embeddings_matrix.shape
        context_half = context_size // 2
        training_data = []
        for sentence in preprocessed_sentences:
            for target_idx, target_word in enumerate(sentence):
                context = []
                for context_idx in range(target_idx - context_half, target_idx + context_half + 1):
                    if context_idx != target_idx and 0 <= context_idx < len(sentence):
                        context.append(word_to_index[sentence[context_idx]])
                if context:
                    training_data.append((word_to_index[target_word], context))

        learning_rate = 0.01
        num_epochs = 100

        for _ in range(num_epochs):
            for target_word_idx, context_words_idx in training_data:
                predicted_vector = np.mean(embeddings_matrix[context_words_idx], axis=0)
                error = embeddings_matrix[target_word_idx] - predicted_vector
                embeddings_matrix[target_word_idx] -= learning_rate * error
                for context_idx in context_words_idx:
                    embeddings_matrix[context_idx] -= learning_rate * error / len(context_words_idx)
        return embeddings_matrix

class SkipgramTrainer:
    def train_skipgram(self, preprocessed_sentences: List[List[str]], word_to_index: Dict[str, int], embeddings_matrix: np.ndarray, context_size: int = 2) -> np.ndarray:
        vocab_size, embedding_dim = embeddings_matrix.shape
        training_data = []
        for sentence in preprocessed_sentences:
            for target_idx, target_word in enumerate(sentence):
                context = []
                for context_idx in range(target_idx - context_size, target_idx + context_size + 1):
                    if context_idx != target_idx and 0 <= context_idx < len(sentence):
                        context.append(word_to_index[sentence[context_idx]])
                for context_idx in context:
                    training_data.append((word_to_index[target_word], context_idx))

        learning_rate = 0.01
        num_epochs = 100

        for _ in range(num_epochs):
            for target_word_idx, context_word_idx in training_data:
                predicted_vector = embeddings_matrix[context_word_idx]
                error = embeddings_matrix[target_word_idx] - predicted_vector
                embeddings_matrix[target_word_idx] -= learning_rate * error
        return embeddings_matrix



# Example usage
#if __name__ == "__main__":
    #dataset_generator = DatasetGenerator()
    #preprocessor = Preprocessor()
    #embeddings_generator = EmbeddingsGenerator()
    #skipgram_trainer = SkipgramTrainer()

    #topic = "Artificial Intelligence"
    #sentences = dataset_generator.generate_sentences(topic, num_sentences=20)
    #preprocessed_sentences = preprocessor.preprocess_sentences(sentences)

    #vocab_size, word_to_index, embeddings_matrix = embeddings_generator.create_word_embeddings(preprocessed_sentences)
    #updated_embeddings_matrix = skipgram_trainer.train_skipgram(preprocessed_sentences, word_to_index, embeddings_matrix)
    #print("Skip-gram embeddings updated.")

    # Test embeddings by finding similar words
    #target_word = "intelligence"
    #target_word_idx = word_to_index[target_word]
    #target_embedding = updated_embeddings_matrix[target_word_idx]

    # Find most similar words
    #similarity_scores = np.dot(updated_embeddings_matrix, target_embedding)
    #similar_indices = np.argsort(similarity_scores)[::-1]

    #print(f"Words similar to '{target_word}':")
    #for idx in similar_indices[:10]:
        #similar_word = [word for word, index in word_to_index.items() if index == idx][0]
        #print(similar_word)

