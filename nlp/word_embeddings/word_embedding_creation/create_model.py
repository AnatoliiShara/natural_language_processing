import itertools

import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from scipy import sparse
from utils import text_preprocess, create_unique_word_dict

#read text from input
texts = pd.read_csv(r'https://github.com/Eligijus112/word-embedding-creation/blob/master/input/sample.csv')
texts = [x for x in texts['text']]

# define window for context
window = 2
# create a placeholder for scanning of the wordlist
word_lists = []
all_text = []

for text in texts:
    # clean text
    text = text_preprocess(text)
    all_text += text
    # create a context dict
    for i, word in enumerate(text):
        for w in range(window):
            # get context that is ahead by "window"  words
            if i + 1 + w < len(text):
                word_lists.append([word] + [text[(i + 1 + w)]])
            # get context that's behind by "window" words
            if i - 1 - w >= 0:
                word_lists.append([word] + [text[(i - w - 1)]])
    unique_word_dict = create_unique_word_dict(all_text)

    # define number of features(unique words)
    n_words = len(unique_word_dict)
    # get all unique words
    words = list(unique_word_dict.keys())

    # create X, Y matrices using one-hot encoding
    X = []
    Y = []
    for i, word_list in tqdm(enumerate(word_lists)):
        # get indices
        main_word_index = unique_word_dict.get(word_list[0])
        context_word_index = unique_word_dict.get(word_list[1])
        X_row = np.zeros(n_words)
        Y_row = np.zeros(n_words)
        # one-hot encode the main word
        X_row[main_word_index] = 1
        # one hot encode Y matrix words
        Y_row[context_word_index] = 1
        X.append(X_row)
        Y.append(Y_row)
    # convert matrices into a sparse format because vast majority of data are 0s
    X = sparse.csr_matrix(X)
    Y = sparse.csr_matrix(Y)

    # define the size of embedding
    embed_size = 2
    # define neural network
    inp = Input(shape=(X.shape[1],))
    x = Dense(units=embed_size, activation='linear')(inp)
    y = Dense(units=Y.shape[1], activation='linear')(x)
    model = Model(input_dim=inp, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # optimize networks params
    model.fit(x=X, y=Y, batch_size=256, epochs=1000)
    weights = model.get_weights()[0]
    # create dict to store embeddings inside. Key - unique word, value - numeric vector
    embedding_dict = {}
    for word in words:
        embedding_dict.update({word: weights[unique_word_dict.get(word)]})

    # Ploting the embeddings
    plt.figure(figsize=(10, 10))
    for word in list(unique_word_dict.keys()):
        coord = embedding_dict.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))
        # Saving the embedding vector to a txt file
    try:
        os.mkdir(f'{os.getcwd()}\\output')
    except Exception as e:
        print(f'Cannot create output folder: {e}')

    with open(f'{os.getcwd()}\\output\\embedding.txt', 'w') as f:
        for key, value in embedding_dict.items():
            try:
                f.write(f'{key}: {value}\n')
            except Exception as e:
                print(f'Cannot write word {key} to dict: {e}')