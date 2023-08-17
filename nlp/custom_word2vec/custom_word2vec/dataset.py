# custom_word2vec/dataset.py

import random

class DatasetGenerator:
    def __init__(self):
        self.topics = topics = {
            "Artificial Intelligence": [
            "Artificial intelligence is transforming industries.",
            "Machine learning is a subset of AI.",
            "Neural networks are used for deep learning.",
            "Natural language processing is a key AI application.",
            "AI has the potential to automate various tasks.",
            "Chatbots are an example of AI in customer service.",
            "AI can analyze large datasets for insights.",
            "Self-driving cars use AI algorithms for navigation.",
            "AI-powered virtual assistants are becoming more common.",
            "AI ethics is an important topic in technology.",
            "Deep learning models can achieve human-like performance.",
            "AI research is advancing rapidly.",
            "Reinforcement learning is used in robotics.",
            "AI can diagnose medical conditions from images.",
            "AI is used in recommendation systems.",
            "Generative models can create realistic images.",
            "AI has the potential to revolutionize healthcare.",
            "AI can be used for fraud detection in finance.",
            "AI algorithms can play complex games.",
            "AI interprets patterns in data to make predictions."
        ]
        # Add more topics and sentences as needed
    }
    def generate_sentences(self, topic, num_sentences):
        if topic in self.topics:
            selected_topic_sentences = self.topics[topic]
            if num_sentences > len(selected_topic_sentences):
                return selected_topic_sentences
            else:
                return random.sample(selected_topic_sentences, num_sentences)
        else:
            raise ValueError("Topic not found in dataset.")

# Example usage
#if __name__ == "__main__":
    #topic = "Artificial Intelligence"
    #sentences = generate_sentences(topic, num_sentences=20)
    #for i, sentence in enumerate(sentences, start=1):
        #print(f"{i}. {sentence}")
