import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from bertopic import Bertopic
from summarization import extractive_summarization

def topic_modelin_nmf(df, num_topics):
    # create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    # fit and transform text data
    tfidf_matrix = vectorizer.fit_transform(df_extractive_summaries['Summary'])
    # perform NMF modeling
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_topics = nmf_model.fit_transform(tfidf_matrix)
    nmf_topics_labels = nmf_topics.argmax(axis=1)
    # add topic label to DataFrame
    df['NMF_Topic'] = nmf_topics_labels
    return df

def topic_modeling_lda(df, num_topics):
    # create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    # fit and transform text data
    tfidf_matrix = vectorizer.fit_transform(df_extractive_summaries['Summary'])
    # Perform LDA topic modeling
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_topics = lda_model.fit_transform(tfidf_matrix)
    lda_topics_labels = lda_topics.argmax(axis=1)
    df["LDA Topic"] = lda_topics_labels
    return df

def topic_modeling_bertopic(df, numtopics):
    # Perform BERTopic topic modeling
    bertopic_model = Bertopic(ngram=(1,2), calculate_probabilities=True)
    bertopic_topics, _ = bertopic_model.fit_transform(df_extractive_summaries['Summary'])
    df['Bertopic'] = bertopic_topics
    return df

# call the topic modeling functions on extractive summaries DataFrame
num_topics = 3
df_extractive_summaries = topic_modelin_nmf(extractive_summarization, num_topics)
df_extractive_summaries = topic_modelin_lda(extractive_summarization, num_topics)
df_extractive_summaries = topic_modeling_bertopic(extractive_summarization, num_topics)

