import pandas as pd
import spacy

# Example texts
texts = {
    'Politics': [
        "Political decisions shape the direction of a country and its governance.",
        "International relations encompass the interactions and relationships between nations on a global scale."
    ],
    'International Relations': [
        "The United Nations serves as a platform for dialogue and cooperation among nations.",
        "Bilateral relations between countries shape foreign policies and impact international trade, security, and cultural exchange."
    ],
    'Sports': [
        "Sports play a crucial role in promoting physical and mental well-being.",
        "The Olympic Games represent the pinnacle of international sporting events."
    ],
    'Arts': [
        "Artistic expression has been a fundamental part of human culture since ancient times.",
        "Art museums and galleries serve as essential spaces for preserving and showcasing artistic heritage."
    ],
    'Sex': [
        "Sexuality is a natural and integral aspect of human identity and encompasses a broad spectrum of orientations, desires, and expressions.",
        "Promoting acceptance, inclusivity, and respect for diverse sexualities is crucial for fostering a more inclusive society."
    ],
    'Music': [
        "Music is a universal language that transcends cultural boundaries and has the power to evoke emotions, unite communities, and tell stories.",
        "Different genres of music offer unique experiences and reflect the cultural, historical, and social contexts in which they emerged."
    ]
}

# Convert the texts into a DataFrame
df = pd.DataFrame([(topic, '. '.join(topic_texts)) for topic, topic_texts in texts.items()], columns=['Topic', 'Texts'])

# Save the DataFrame as a CSV file
df.to_csv('texts.csv', index=False)

# Read the CSV file using Pandas
df_read = pd.read_csv('texts.csv')

# Print the DataFrame
print(df_read)
print()

# Extractive Summarization
def extractive_summarization(text):
    # load SpaCy English language model
    nlp = spacy.load("en_core_web_sm")
    # tokenize text and split into sents
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # calculate the desired num of sents for summary based on ratio
    num_sentences = int(len(sentences) * 0.2)
    # ensure that num of sentences is at least 1
    num_sentences = max(num_sentences, 1)
    # sort sents based on their length
    sorted_sentences = sorted(sentences, key=len, reverse=True)
    summary = " ".join(sorted_sentences[:num_sentences])
    return summary

# summarize text using extractive method
extractive_summaries = {}
for topic, topic_texts in texts.items():
    summaries = []
    for text in topic_texts:
        summary = extractive_summarization(text)
        summaries.append(summary)
    extractive_summaries[topic] = summaries

# print extractive summaries
for topic, summaries in extractive_summaries.items():
    print(f"Topic: {topic}")
    for i, summary in enumerate(summaries):
        print(f"Summary {i+1}: {summary}")
# collect summaries into DataFrame
summary_data = []
for topic, summaries in extractive_summaries.items():
    for i, summary in enumerate(summaries):
        summary_data.append((topic, i + 1, summary))

df_summaries = pd.DataFrame(summary_data, columns=['Topic', 'Summary Index', 'Summary'])
print(df_summaries)