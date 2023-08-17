import pandas as pd
import nltk
import spacy
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')

# Load SpaCy model for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Load sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Read the CSV file
data = pd.read_csv('examiner-date-text.csv')

# Extract headlines from the CSV data
corpus = data['headline_text'].tolist()

# Analyze Named Entities
def contains_named_entities(headline):
    doc = nlp(headline)
    return any(ent.label_ in ['PERSON', 'ORG'] for ent in doc.ents)

named_entity_headlines = [headline for headline in corpus if contains_named_entities(headline)]
percentage_named_entity = (len(named_entity_headlines) / len(corpus)) * 100

# Analyze Emotional Coloring
def has_positive_emotion(headline):
    sentiment_scores = sia.polarity_scores(headline)
    return sentiment_scores['compound'] > 0

positive_emotion_headlines = [headline for headline in corpus if has_positive_emotion(headline)]
percentage_positive_emotion = (len(positive_emotion_headlines) / len(corpus)) * 100

# Analyze Degrees of Comparison and Excellence
def contains_superlative_or_excellence(headline):
    positive_count = 0
    negative_count = 0

    for word in word_tokenize(headline):
        synsets = swn.senti_synsets(word)
        if synsets:
            avg_pos = sum(s.pos_score() for s in synsets) / len(synsets)
            avg_neg = sum(s.neg_score() for s in synsets) / len(synsets)
            if avg_pos > avg_neg and (avg_pos - avg_neg) > 0.2:
                positive_count += 1
            elif avg_neg > avg_pos and (avg_neg - avg_pos) > 0.2:
                negative_count += 1

    return positive_count + negative_count > 0

superlative_excellence_headlines = [headline for headline in corpus if contains_superlative_or_excellence(headline)]
percentage_superlative_excellence = (len(superlative_excellence_headlines) / len(corpus)) * 100

# Save the results
statistics = {
    "Percentage of headlines with named entities": percentage_named_entity,
    "Percentage of headlines with positive emotion": percentage_positive_emotion,
    "Percentage of headlines with superlatives/excellence": percentage_superlative_excellence,
}

with open("your_name_statistics.txt", "w") as f:
    for feature, percentage in statistics.items():
        f.write(f"{feature}: {percentage:.2f}%\n")

print("Analysis completed and results saved.")

